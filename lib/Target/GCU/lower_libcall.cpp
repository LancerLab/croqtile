#include "lower_libcall.hpp"
#include "topscc_codegen.hpp"

using namespace Choreo;
using namespace Choreo::Topscc;

void TopsccCodeGen::EmitLibCall(AST::Call& n, const std::string& func_name,
                                std::ostringstream& os,
                                const std::string& indent) {
  auto kind = ClassifyLibCall(func_name);
  auto acore_name = LibCallToAcoreName(func_name);

  auto DeduceElemCast = [&](int arg_idx) -> std::string {
    auto ty = n.arguments->ValueAt(arg_idx)->GetType();
    if (!ty) return "";
    BaseType bt = BaseType::UNKNOWN;
    if (auto sty = dyn_cast<SpannedType>(ty))
      bt = sty->ElementType();
    else if (auto dt = dyn_cast<DeviceDataType>(ty))
      bt = Choreo::GetBaseType(*dt);
    else
      bt = ty->GetBaseType();
    if (bt == BaseType::UNKNOWN) return "";
    if (bt == BaseType::BF16) return "(__bf16*)";
    if (bt == BaseType::F16) return "(__fp16*)";
    return std::string("(") + NameBaseType(bt, false) + "*)";
  };

  auto DeduceBaseType = [&](int arg_idx) -> BaseType {
    auto ty = n.arguments->ValueAt(arg_idx)->GetType();
    if (!ty) return BaseType::UNKNOWN;
    if (auto sty = dyn_cast<SpannedType>(ty))
      return sty->ElementType();
    else if (auto dt = dyn_cast<DeviceDataType>(ty))
      return Choreo::GetBaseType(*dt);
    return ty->GetBaseType();
  };

  if (kind == LibCallKind::GEMM) {
    auto argc = n.arguments->Count();
    bool has_bias = (argc == 6);

    size_t out_idx = 0, a_idx = 1, b_idx = 2;
    size_t bias_idx = has_bias ? 3 : SIZE_MAX;
    size_t k_idx = has_bias ? 4 : 3;
    size_t n_idx = has_bias ? 5 : 4;

    std::string out_str = ExprSTR(n.arguments->ValueAt(out_idx), false);
    std::string a_str = ExprSTR(n.arguments->ValueAt(a_idx), false);
    std::string b_str = ExprSTR(n.arguments->ValueAt(b_idx), false);
    std::string k_str = ExprSTR(n.arguments->ValueAt(k_idx), false);
    std::string n_str = ExprSTR(n.arguments->ValueAt(n_idx), false);
    BaseType lhs_bt = DeduceBaseType(a_idx);

    std::string bias_str;
    if (has_bias)
      bias_str = ExprSTR(n.arguments->ValueAt(bias_idx), false);
    else if (lhs_bt != BaseType::UNKNOWN)
      bias_str = std::string("(") + NameBaseType(lhs_bt, false) + "*)nullptr";
    else
      bias_str = "(void*)nullptr";

    if (CCtx().UseTargetLib()) {
      auto warn_fn = [](const auto&, const std::string&) {};
      auto assess_fn = [](const std::string&, AST::Node*) {};
      auto info = AnalyzeLibGemm(n, assess_fn, warn_fn);

      if (info.use_acore) {
        has_acore_call = true;
        os << indent << "{ // __lib_gemm -> acore::matmul\n";
        os << indent << "  int __lib_gemm_ws__[2048];\n";

        if (info.use_static_M) {
          os << indent << "  acore::matmul<" << info.static_M
             << ", acore::MK_KN>(" << out_str << ", " << a_str << ", " << b_str
             << ", " << bias_str << ", __lib_gemm_ws__, " << k_str << ", "
             << n_str << ", 0, 1, " << (has_bias ? "1" : "0") << ", 0, 0);\n";
        } else {
          auto out_ty = n.arguments->ValueAt(out_idx)->GetType();
          std::string m_str = std::to_string(info.static_M);
          if (out_ty)
            if (auto sty = dyn_cast<SpannedType>(out_ty)) {
              auto shape = sty->GetShape();
              if (shape.Rank() >= 1) m_str = ValueSTR(shape.Value()[0]);
            }
          os << indent << "  acore::matmul<acore::MK_KN>(" << out_str << ", "
             << a_str << ", " << b_str << ", " << bias_str
             << ", __lib_gemm_ws__, " << m_str << ", " << k_str << ", " << n_str
             << ", 0, 1, " << (has_bias ? "1" : "0") << ", 0, 0);\n";
        }
        os << indent << "}\n";
        return;
      }
    }

    // Fallback to runtime template
    Warning(n.LOC(), "'" + func_name +
                         "' using loop-based general fallback "
                         "(low performance).");
    has_lib_gemm_general = true;
    auto out_ty = n.arguments->ValueAt(out_idx)->GetType();
    std::string m_str = k_str;
    if (out_ty)
      if (auto sty = dyn_cast<SpannedType>(out_ty)) {
        auto shape = sty->GetShape();
        if (shape.Rank() >= 1) m_str = ValueSTR(shape.Value()[0]);
      }

    std::string out_type_str = NameBaseType(
        lhs_bt == BaseType::UNKNOWN ? BaseType::F32 : lhs_bt, true);
    if (has_bias)
      os << indent << "choreo::lib_gemm_bias_general<" << out_type_str << ">("
         << out_str << ", " << a_str << ", " << b_str << ", " << bias_str
         << ", " << m_str << ", " << k_str << ", " << n_str << ");\n";
    else
      os << indent << "choreo::lib_gemm_general<" << out_type_str << ">("
         << out_str << ", " << a_str << ", " << b_str << ", " << m_str << ", "
         << k_str << ", " << n_str << ");\n";

  } else if (kind == LibCallKind::UNARY) {
    std::string dst = ExprSTR(n.arguments->ValueAt(0), false);
    std::string src = ExprSTR(n.arguments->ValueAt(1), false);
    std::string num = ExprSTR(n.arguments->ValueAt(2), false);

    if (CCtx().UseTargetLib()) {
      has_acore_call = true;
      auto cast = DeduceElemCast(0);
      os << indent << "acore::" << acore_name << "(" << cast << dst << ", "
         << cast << src << ", " << num << ");\n";
    } else {
      Warning(n.LOC(), "'" + func_name +
                           "' using loop-based general fallback "
                           "(low performance).");
      has_lib_fallback = true;
      auto bt = DeduceBaseType(0);
      std::string type_str =
          NameBaseType(bt == BaseType::UNKNOWN ? BaseType::F32 : bt, true);
      os << indent << "choreo::lib_" << acore_name << "<" << type_str << ">("
         << dst << ", " << src << ", " << num << ");\n";
    }

  } else if (kind == LibCallKind::BINARY) {
    std::string dst = ExprSTR(n.arguments->ValueAt(0), false);
    std::string lhs = ExprSTR(n.arguments->ValueAt(1), false);
    std::string rhs = ExprSTR(n.arguments->ValueAt(2), false);
    std::string num = ExprSTR(n.arguments->ValueAt(3), false);

    if (CCtx().UseTargetLib()) {
      has_acore_call = true;
      auto cast = DeduceElemCast(0);
      os << indent << "acore::" << acore_name << "(" << cast << dst << ", "
         << cast << lhs << ", " << cast << rhs << ", " << num << ");\n";
    } else {
      Warning(n.LOC(), "'" + func_name +
                           "' using loop-based general fallback "
                           "(low performance).");
      has_lib_fallback = true;
      auto bt = DeduceBaseType(0);
      std::string type_str =
          NameBaseType(bt == BaseType::UNKNOWN ? BaseType::F32 : bt, true);
      os << indent << "choreo::lib_" << acore_name << "<" << type_str << ">("
         << dst << ", " << lhs << ", " << rhs << ", " << num << ");\n";
    }

  } else if (kind == LibCallKind::ACTIVATION_PARAM) {
    std::string dst = ExprSTR(n.arguments->ValueAt(0), false);
    std::string src = ExprSTR(n.arguments->ValueAt(1), false);
    std::string num = ExprSTR(n.arguments->ValueAt(2), false);
    auto argc = n.arguments->Count();
    std::vector<std::string> extras;
    for (size_t i = 3; i < argc; ++i)
      extras.push_back(ExprSTR(n.arguments->ValueAt(i), false));

    if (CCtx().UseTargetLib()) {
      has_acore_call = true;
      auto cast = DeduceElemCast(0);
      os << indent << "acore::" << acore_name << "(" << cast << dst << ", "
         << cast << src << ", " << num;
      for (auto& e : extras) os << ", " << e;
      os << ");\n";
    } else {
      Warning(n.LOC(), "'" + func_name +
                           "' using loop-based general fallback "
                           "(low performance).");
      has_lib_fallback = true;
      auto bt = DeduceBaseType(0);
      std::string type_str =
          NameBaseType(bt == BaseType::UNKNOWN ? BaseType::F32 : bt, true);
      os << indent << "choreo::lib_" << acore_name << "<" << type_str << ">("
         << dst << ", " << src << ", " << num;
      for (auto& e : extras) os << ", " << e;
      os << ");\n";
    }

  } else if (kind == LibCallKind::ACTIVATION_BIN) {
    std::string dst = ExprSTR(n.arguments->ValueAt(0), false);
    std::string lhs = ExprSTR(n.arguments->ValueAt(1), false);
    std::string rhs = ExprSTR(n.arguments->ValueAt(2), false);
    std::string num = ExprSTR(n.arguments->ValueAt(3), false);

    if (CCtx().UseTargetLib()) {
      has_acore_call = true;
      auto cast = DeduceElemCast(0);
      os << indent << "acore::" << acore_name << "(" << cast << dst << ", "
         << cast << lhs << ", " << cast << rhs << ", " << num << ");\n";
    } else {
      Warning(n.LOC(), "'" + func_name +
                           "' using loop-based general fallback "
                           "(low performance).");
      has_lib_fallback = true;
      auto bt = DeduceBaseType(0);
      std::string type_str =
          NameBaseType(bt == BaseType::UNKNOWN ? BaseType::F32 : bt, true);
      os << indent << "choreo::lib_" << acore_name << "<" << type_str << ">("
         << dst << ", " << lhs << ", " << rhs << ", " << num << ");\n";
    }

  } else if (kind == LibCallKind::REDUCE) {
    std::string dst = ExprSTR(n.arguments->ValueAt(0), false);
    std::string src = ExprSTR(n.arguments->ValueAt(1), false);
    std::string num = ExprSTR(n.arguments->ValueAt(2), false);
    std::string rdim = ExprSTR(n.arguments->ValueAt(3), false);
    std::string nred = ExprSTR(n.arguments->ValueAt(4), false);

    if (CCtx().UseTargetLib()) {
      has_acore_call = true;
      auto cast = DeduceElemCast(0);
      auto bt = DeduceBaseType(0);
      std::string data_type =
          NameBaseType(bt == BaseType::UNKNOWN ? BaseType::F32 : bt, false);
      std::string cal_type = (bt == BaseType::F32 || bt == BaseType::F16 ||
                              bt == BaseType::BF16 || bt == BaseType::UNKNOWN)
                                 ? "float"
                                 : "int";
      std::string init_val = (bt == BaseType::F32 || bt == BaseType::F16 ||
                              bt == BaseType::BF16 || bt == BaseType::UNKNOWN)
                                 ? "0.0f"
                                 : "0";
      // acore::reduce_sum<axis, data_t, cal_t>(dst, src, ws,
      //   init_value, dim2, dim1, dim0, is_first, is_last)
      // axis=0: reduce along dim0; shape = (1, num/nred, nred)
      os << indent << "{ // " << func_name << " -> acore::" << acore_name
         << "\n";
      os << indent << "  __local__ __valigned__ " << cal_type
         << " __reduce_ws__[2048];\n";
      os << indent << "  acore::" << acore_name << "<0, " << data_type << ", "
         << cal_type << ">(" << cast << dst << ", " << cast << src
         << ", __reduce_ws__, (" << data_type << ")" << init_val << ", 1, "
         << num << " / " << nred << ", " << nred << ", true, true);\n";
      os << indent << "}\n";
    } else {
      Warning(n.LOC(), "'" + func_name +
                           "' using loop-based general fallback "
                           "(low performance).");
      has_lib_fallback = true;
      auto bt = DeduceBaseType(0);
      std::string type_str =
          NameBaseType(bt == BaseType::UNKNOWN ? BaseType::F32 : bt, true);
      os << indent << "choreo::lib_" << acore_name << "<" << type_str << ">("
         << dst << ", " << src << ", " << num << ", " << rdim << ", " << nred
         << ");\n";
    }

  } else if (kind == LibCallKind::ADDMM) {
    std::string out_s = ExprSTR(n.arguments->ValueAt(0), false);
    std::string bias_s = ExprSTR(n.arguments->ValueAt(1), false);
    std::string a_s = ExprSTR(n.arguments->ValueAt(2), false);
    std::string b_s = ExprSTR(n.arguments->ValueAt(3), false);
    std::string m_s = ExprSTR(n.arguments->ValueAt(4), false);
    std::string k_s = ExprSTR(n.arguments->ValueAt(5), false);
    std::string n_s = ExprSTR(n.arguments->ValueAt(6), false);
    std::string alpha_s = ExprSTR(n.arguments->ValueAt(7), false);
    std::string beta_s = ExprSTR(n.arguments->ValueAt(8), false);
    BaseType lhs_bt = DeduceBaseType(2);

    if (CCtx().UseTargetLib()) {
      has_acore_call = true;
      os << indent << "{ // __lib_addmm -> acore::addmm\n";
      os << indent << "  int __lib_addmm_ws__[2048];\n";
      os << indent << "  acore::addmm<acore::MK_KN>(" << out_s << ", " << a_s
         << ", " << b_s << ", " << bias_s << ", __lib_addmm_ws__, " << m_s
         << ", " << k_s << ", " << n_s << ", 0, " << alpha_s << ", " << beta_s
         << ");\n";
      os << indent << "}\n";
    } else {
      Warning(n.LOC(), "'" + func_name +
                           "' using loop-based general fallback "
                           "(low performance).");
      has_lib_fallback = true;
      std::string type_str = NameBaseType(
          lhs_bt == BaseType::UNKNOWN ? BaseType::F32 : lhs_bt, true);
      os << indent << "choreo::lib_addmm<" << type_str << ">(" << out_s << ", "
         << bias_s << ", " << a_s << ", " << b_s << ", " << m_s << ", " << k_s
         << ", " << n_s << ", " << alpha_s << ", " << beta_s << ");\n";
    }

  } else if (kind == LibCallKind::NORM) {
    std::string dst = ExprSTR(n.arguments->ValueAt(0), false);
    std::string src = ExprSTR(n.arguments->ValueAt(1), false);
    std::string weight = ExprSTR(n.arguments->ValueAt(2), false);
    std::string bias = ExprSTR(n.arguments->ValueAt(3), false);
    std::string batch = ExprSTR(n.arguments->ValueAt(4), false);
    std::string norm_size = ExprSTR(n.arguments->ValueAt(5), false);
    std::string eps = ExprSTR(n.arguments->ValueAt(6), false);

    if (CCtx().UseTargetLib()) {
      has_acore_call = true;
      auto cast = DeduceElemCast(0);
      os << indent << "acore::layer_norm(" << cast << dst << ", " << cast << src
         << ", " << cast << weight << ", " << cast << bias << ", " << batch
         << ", " << norm_size << ", " << eps << ");\n";
    } else {
      Warning(n.LOC(), "'" + func_name +
                           "' using loop-based general fallback "
                           "(low performance).");
      has_lib_fallback = true;
      auto bt = DeduceBaseType(0);
      std::string type_str =
          NameBaseType(bt == BaseType::UNKNOWN ? BaseType::F32 : bt, true);
      os << indent << "choreo::lib_layer_norm<" << type_str << ">(" << dst
         << ", " << src << ", " << weight << ", " << bias << ", " << batch
         << ", " << norm_size << ", " << eps << ");\n";
    }

  } else if (kind == LibCallKind::CONV2D) {
    std::string out_s = ExprSTR(n.arguments->ValueAt(0), false);
    std::string inp_s = ExprSTR(n.arguments->ValueAt(1), false);
    std::string wgt_s = ExprSTR(n.arguments->ValueAt(2), false);
    std::string bias_s = ExprSTR(n.arguments->ValueAt(3), false);
    std::string batch_s = ExprSTR(n.arguments->ValueAt(4), false);
    std::string hi_s = ExprSTR(n.arguments->ValueAt(5), false);
    std::string wi_s = ExprSTR(n.arguments->ValueAt(6), false);
    std::string ci_s = ExprSTR(n.arguments->ValueAt(7), false);
    std::string co_s = ExprSTR(n.arguments->ValueAt(8), false);
    std::string kh_s = ExprSTR(n.arguments->ValueAt(9), false);
    std::string kw_s = ExprSTR(n.arguments->ValueAt(10), false);
    std::string sh_s = ExprSTR(n.arguments->ValueAt(11), false);
    std::string sw_s = ExprSTR(n.arguments->ValueAt(12), false);
    std::string ph_s = ExprSTR(n.arguments->ValueAt(13), false);
    std::string pw_s = ExprSTR(n.arguments->ValueAt(14), false);

    if (CCtx().UseTargetLib()) {
      has_acore_call = true;
      os << indent << "{ // __lib_conv2d -> acore::conv2d\n";
      os << indent << "  int __ho = (" << hi_s << " + 2*" << ph_s << " - "
         << kh_s << ") / " << sh_s << " + 1;\n";
      os << indent << "  int __wo = (" << wi_s << " + 2*" << pw_s << " - "
         << kw_s << ") / " << sw_s << " + 1;\n";
      os << indent << "  acore::conv2d<__ho, __wo>(" << out_s << ", " << inp_s
         << ", " << wgt_s << ", " << bias_s << ", " << batch_s << ", " << hi_s
         << ", " << wi_s << ", " << ci_s << ", " << kh_s << ", " << kw_s << ", "
         << co_s << ", __ho, __wo, " << sh_s << ", " << sw_s
         << ", 1, 1, 1, 0, 0);\n";
      os << indent << "}\n";
    } else {
      Warning(n.LOC(), "'" + func_name +
                           "' using loop-based general fallback "
                           "(low performance).");
      has_lib_fallback = true;
      auto bt = DeduceBaseType(0);
      std::string type_str =
          NameBaseType(bt == BaseType::UNKNOWN ? BaseType::F32 : bt, true);
      os << indent << "choreo::lib_conv2d<" << type_str << ">(" << out_s << ", "
         << inp_s << ", " << wgt_s << ", " << bias_s << ", " << batch_s << ", "
         << hi_s << ", " << wi_s << ", " << ci_s << ", " << co_s << ", " << kh_s
         << ", " << kw_s << ", " << sh_s << ", " << sw_s << ", " << ph_s << ", "
         << pw_s << ");\n";
    }

  } else if (kind == LibCallKind::CONVERT) {
    std::string dst = ExprSTR(n.arguments->ValueAt(0), false);
    std::string src = ExprSTR(n.arguments->ValueAt(1), false);
    std::string num = ExprSTR(n.arguments->ValueAt(2), false);

    if (CCtx().UseTargetLib()) {
      has_acore_call = true;
      auto dst_cast = DeduceElemCast(0);
      auto src_cast = DeduceElemCast(1);
      os << indent << "acore::convert(" << dst_cast << dst << ", " << src_cast
         << src << ", " << num << ");\n";
    } else {
      Warning(n.LOC(), "'" + func_name +
                           "' using loop-based general fallback "
                           "(low performance).");
      has_lib_fallback = true;
      auto dst_bt = DeduceBaseType(0);
      auto src_bt = DeduceBaseType(1);
      std::string dst_ts = NameBaseType(
          dst_bt == BaseType::UNKNOWN ? BaseType::F32 : dst_bt, true);
      std::string src_ts = NameBaseType(
          src_bt == BaseType::UNKNOWN ? BaseType::F32 : src_bt, true);
      os << indent << "choreo::lib_convert<" << dst_ts << ", " << src_ts << ">("
         << dst << ", " << src << ", " << num << ");\n";
    }

  } else if (kind == LibCallKind::POINTWISE) {
    std::string dst = ExprSTR(n.arguments->ValueAt(0), false);
    std::string arg1 = ExprSTR(n.arguments->ValueAt(1), false);
    std::string arg2 = ExprSTR(n.arguments->ValueAt(2), false);
    std::string arg3 = ExprSTR(n.arguments->ValueAt(3), false);
    std::string num = ExprSTR(n.arguments->ValueAt(4), false);

    if (CCtx().UseTargetLib()) {
      has_acore_call = true;
      auto cast = DeduceElemCast(0);
      os << indent << "acore::" << acore_name << "(" << cast << dst << ", "
         << arg1 << ", " << cast << arg2 << ", " << cast << arg3 << ", " << num
         << ");\n";
    } else {
      Warning(n.LOC(), "'" + func_name +
                           "' using loop-based general fallback "
                           "(low performance).");
      has_lib_fallback = true;
      auto bt = DeduceBaseType(0);
      std::string type_str =
          NameBaseType(bt == BaseType::UNKNOWN ? BaseType::F32 : bt, true);
      os << indent << "choreo::lib_" << acore_name << "<" << type_str << ">("
         << dst << ", " << arg1 << ", " << arg2 << ", " << arg3 << ", " << num
         << ");\n";
    }

  } else {
    choreo_unreachable("'" + func_name +
                       "' is not a supported __lib_ builtin. "
                       "No fallback implementation available.");
  }
}
