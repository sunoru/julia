// This file is a part of Julia. License is MIT: https://julialang.org/license

// Function multi-versioning
#define DEBUG_TYPE "julia_mv"
#undef DEBUG

// LLVM pass to clone function for different archs

#include "llvm-version.h"
#include "support/dtypes.h"

#include <llvm/Pass.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Analysis/LoopInfo.h>
#if JL_LLVM_VERSION >= 30700
#include <llvm/IR/LegacyPassManager.h>
#else
#include <llvm/PassManager.h>
#endif
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include "fix_llvm_assert.h"

#include "julia.h"
#include "julia_internal.h"

#include <unordered_map>
#include <vector>

using namespace llvm;

extern std::pair<MDNode*,MDNode*> tbaa_make_child(const char *name, MDNode *parent=nullptr, bool isConstant=false);
extern "C" void jl_dump_llvm_value(void *v);

namespace {

struct JuliaMV: public ModulePass {
    static char ID;
    JuliaMV()
        : ModulePass(ID)
    {}

private:
    bool runOnModule(Module &M) override;
    void getAnalysisUsage(AnalysisUsage &AU) const override
    {
        AU.addRequired<LoopInfoWrapperPass>();
        AU.setPreservesAll();
    }
    bool shouldClone(Function &F);
    bool checkUses(Function &F, Constant *fary);
    bool checkUses(Function &F, Constant *V, Constant *fary, bool &inFVars);
    bool checkConstantUse(Function &F, Constant *V, Constant *fary, bool &inFVars);
};

bool JuliaMV::shouldClone(Function &F)
{
    if (F.empty())
        return false;
    auto &LI = getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
    if (!LI.empty())
        return true;
    for (auto &bb: F) {
        for (auto &I: bb) {
            if (auto call = dyn_cast<CallInst>(&I)) {
                if (auto callee = call->getCalledFunction()) {
                    auto name = callee->getName();
                    if (name.startswith("llvm.muladd.") || name.startswith("llvm.fma.")) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

bool JuliaMV::checkUses(Function &F, Constant *fary)
{
    bool inFVars = false;
    bool res = checkUses(F, &F, fary, inFVars);
    return res && inFVars;
}

bool JuliaMV::checkConstantUse(Function &F, Constant *V, Constant *fary, bool &inFVars)
{
    if (V == fary) {
        inFVars = true;
        return true;
    }
    if (auto cexpr = dyn_cast<ConstantExpr>(V)) {
        if (cexpr->getOpcode() == Instruction::BitCast) {
            return checkUses(F, V, fary, inFVars);
        }
    }
    return false;
}

bool JuliaMV::checkUses(Function &F, Constant *V, Constant *fary, bool &inFVars)
{
    for (auto *user: V->users()) {
        if (isa<Instruction>(user))
            continue;
        auto *C = dyn_cast<Constant>(user);
        if (!C || !checkConstantUse(F, C, fary, inFVars)) {
            return false;
        }
    }
    return true;
}

static Function *getFunction(Value *v)
{
    if (auto f = dyn_cast<Function>(v))
        return f;
    if (auto c = dyn_cast<ConstantExpr>(v)) {
        if (c->getOpcode() == Instruction::BitCast) {
            return getFunction(c->getOperand(0));
        }
    }
    return nullptr;
}

static void addFeatures(Function *F)
{
    auto attr = F->getFnAttribute("target-features");
    std::string feature =
        "+avx2,+avx,+fma,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3";
    if (attr.isStringAttribute()) {
        feature += ",";
        feature += attr.getValueAsString();
    }
    F->addFnAttr("target-features", feature);
}

bool JuliaMV::runOnModule(Module &M)
{
    MDNode *tbaa_const = tbaa_make_child("jtbaa_const", nullptr, true).first;
    GlobalVariable *fvars = M.getGlobalVariable("jl_sysimg_fvars");
    // This makes sure this only runs during sysimg generation
    if (!fvars || !fvars->hasInitializer())
        return true;
    auto *fary = dyn_cast<ConstantArray>(fvars->getInitializer());
    if (!fary)
        return true;
    LLVMContext &ctx = M.getContext();
    ValueToValueMapTy VMap;
    for (auto &F: M) {
        if (shouldClone(F) && checkUses(F, fary)) {
            Function *NF = Function::Create(cast<FunctionType>(F.getValueType()),
                                            F.getLinkage(), F.getName() + ".avx2", &M);
            NF->copyAttributesFrom(&F);
            VMap[&F] = NF;
        }
    }
    std::unordered_map<Function*,size_t> idx_map;
    size_t nf = fary->getNumOperands();
    for (size_t i = 0; i < nf; i++) {
        if (Function *ele = getFunction(fary->getOperand(i))) {
            auto it = VMap.find(ele);
            if (it != VMap.end()) {
                idx_map[ele] = i;
            }
        }
    }
    for (auto I: idx_map) {
        auto oldF = I.first;
        auto newF = cast<Function>(VMap[oldF]);
        Function::arg_iterator DestI = newF->arg_begin();
        for (Function::const_arg_iterator J = oldF->arg_begin(); J != oldF->arg_end(); ++J) {
            DestI->setName(J->getName());
            VMap[&*J] = &*DestI++;
        }
        SmallVector<ReturnInst*,8> Returns;
        CloneFunctionInto(newF, oldF, VMap, false, Returns);
        addFeatures(newF);
    }
    std::vector<Constant*> ptrs;
    std::vector<Constant*> idxs;
    auto T_void = Type::getVoidTy(ctx);
    auto T_pvoidfunc = FunctionType::get(T_void, false)->getPointerTo();
    auto T_size = (sizeof(size_t) == 8 ? Type::getInt64Ty(ctx) : Type::getInt32Ty(ctx));
    for (auto I: idx_map) {
        auto oldF = I.first;
        auto idx = I.second;
        auto newF = cast<Function>(VMap[oldF]);
        ptrs.push_back(ConstantExpr::getBitCast(newF, T_pvoidfunc));
        auto offset = ConstantInt::get(T_size, idx);
        idxs.push_back(offset);
        for (auto user: oldF->users()) {
            auto inst = dyn_cast<Instruction>(user);
            if (!inst)
                continue;
            auto encloseF = inst->getParent()->getParent();
            if (VMap.find(encloseF) != VMap.end())
                continue;
            Value *slot = ConstantExpr::getBitCast(fvars, T_pvoidfunc->getPointerTo());
            slot = GetElementPtrInst::Create(T_pvoidfunc, slot, {offset}, "", inst);
            Instruction *ptr = new LoadInst(slot, "", inst);
            ptr->setMetadata(llvm::LLVMContext::MD_tbaa, tbaa_const);
            ptr = new BitCastInst(ptr, oldF->getType(), "", inst);
            inst->replaceUsesOfWith(oldF, ptr);
        }
    }
    ArrayType *fvars_type = ArrayType::get(T_pvoidfunc, ptrs.size());
    auto ptr_gv = new GlobalVariable(M, fvars_type, true, GlobalVariable::InternalLinkage,
                                     ConstantArray::get(fvars_type, ptrs));
    ArrayType *idxs_type = ArrayType::get(T_size, idxs.size());
    auto idx_gv = new GlobalVariable(M, idxs_type, true, GlobalVariable::InternalLinkage,
                                     ConstantArray::get(idxs_type, idxs));

    // TODO
    std::vector<Type*> dispatch_args(0);
    dispatch_args.push_back(T_size); // hasavx2
    dispatch_args.push_back(T_size->getPointerTo());
    dispatch_args.push_back(fvars_type->getPointerTo()->getPointerTo());
    dispatch_args.push_back(idxs_type->getPointerTo()->getPointerTo());
    Function *dispatchF = Function::Create(FunctionType::get(T_void, dispatch_args, false),
                                           Function::ExternalLinkage,
                                           "jl_dispatch_sysimg_fvars", &M);
    IRBuilder<> builder(ctx);
    BasicBlock *b0 = BasicBlock::Create(ctx, "top", dispatchF);
    builder.SetInsertPoint(b0);
    DebugLoc noDbg;
    builder.SetCurrentDebugLocation(noDbg);

    std::vector<Argument*> args;
    for (auto &arg: dispatchF->args())
        args.push_back(&arg);

    auto sz_arg = args[1];
    auto fvars_arg = args[2];
    auto idxs_arg = args[3];

    builder.CreateStore(ConstantInt::get(T_size, ptrs.size()), sz_arg);

    BasicBlock *match_bb = BasicBlock::Create(ctx, "match");
    BasicBlock *fail_bb = BasicBlock::Create(ctx, "fail");
    builder.CreateCondBr(builder.CreateICmpEQ(args[0], ConstantInt::get(T_size, 1)),
                         match_bb, fail_bb);

    dispatchF->getBasicBlockList().push_back(match_bb);
    builder.SetInsertPoint(match_bb);
    builder.CreateStore(ptr_gv, fvars_arg);
    builder.CreateStore(idx_gv, idxs_arg);
    builder.CreateRetVoid();

    dispatchF->getBasicBlockList().push_back(fail_bb);
    builder.SetInsertPoint(fail_bb);
    builder.CreateStore(ConstantPointerNull::get(fvars_type->getPointerTo()), fvars_arg);
    builder.CreateStore(ConstantPointerNull::get(idxs_type->getPointerTo()), idxs_arg);
    builder.CreateRetVoid();

    return true;
}

char JuliaMV::ID = 0;
static RegisterPass<JuliaMV> X("JuliaMV", "JuliaMV Pass",
                               false /* Only looks at CFG */,
                               false /* Analysis Pass */);

}

Pass *createJuliaMVPass()
{
    return new JuliaMV();
}
