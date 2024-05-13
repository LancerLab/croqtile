" Quit if a syntax file has already been loaded
if exists("b:current_syntax")
  finish
endif

" Syntax highlighting rules
syntax clear

" Inherits C++ syntax highlighting as a base
runtime! syntax/cpp.vim

" Define custom syntax elements
syntax keyword coKeyword chunkAt mdspan with parallel ituple by in foreach shared local global where
syntax match coType "\<\(f32\|f16\|bf16\|u32\|s32\|u16\|s16\|u8\|s8\)\>"
syntax match coAttribute "\(__co__\|__cok__\)"
syntax match coOperator "\(=>\|dcb\|#)"
syntax match coFunction "\<\(wait\|call\|dte\.reshape\|dte\.linear\|dte\.slice\|dte\.deslice\)\>"

highlight coOperator guifg=cyan ctermfg=cyan gui=bold

" Link custom syntax elements to existing highlighting groups
highlight link coKeyword Keyword
highlight link coType Type
highlight link coAttribute PreProc
highlight link coFunction Function
highlight link coOperator Operator

let b:current_syntax = "co"
