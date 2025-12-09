" Quit if a syntax file has already been loaded

if exists("b:current_syntax")
  finish
endif

" Syntax highlighting rules
syntax clear

" Inherits C++ syntax highlighting as a base
runtime! syntax/cpp.vim

" Define custom syntax elements
syntax keyword coKeyword mdspan with parallel ituple by in foreach shared local global where after inthreads event
syntax match coType "\<\(f32\|f16\|bf16\|f8\|u64\|s64\|u32\|s32\|u16\|s16\|u8\|s8\|half8\|half\|bfp16\)\>"
syntax match coAttribute "\(__co__\|__cok__\)"
syntax match coOperator "\(=>\|#\|cdiv\)"
syntax match coFunction "\(\<wait\>\|\<trigger\>\|\<call\>\|\<select\>\|\<swap\>\|\<rotate\>\|\<sync\.\(shared\|global\|local\)\>\|\.\<async\>\|\.span_as\|\.chunkat\|\.chunk\|\.subspan\|\.modspan\|\.stride\|\.at\|\<\(dma\|tma\)\.\(any\|copy\|transp\|pad\)\>\(\.async\)\?\|\<mma\.\(fill\|load\|store\|row\.col\|row\.row\|col\.row\|col\.col\)\>\(\.async\)\?\)"

highlight coOperator guifg=cyan ctermfg=cyan gui=bold

" Link custom syntax elements to existing highlighting groups
highlight link coKeyword Keyword
highlight link coType Type
highlight link coAttribute PreProc
highlight link coFunction Function
highlight link coOperator Operator

let b:current_syntax = "co"

