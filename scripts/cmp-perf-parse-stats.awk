# cmp-perf-parse-stats.awk -- parse PrintAssessmentStats stderr output
# into key=value pairs for consumption by cmp-perf-stats.sh.
#
# Usage: awk -f cmp-perf-parse-stats.awk < stats_output.txt
# Output format: KEY=VALUE (one per line)

BEGIN {
  in_section = 0
  sep_seen = 0
}

# Detect the beginning of the Assessment Statistics section
/===----/ && sep_seen == 0 {
  sep_seen = 1
  next
}
/Assessment Statistics/ && sep_seen == 1 {
  sep_seen = 2
  next
}
/===----/ && sep_seen == 2 {
  in_section = 1
  next
}

# Exit when section ends (blank line after stats block,
# or the next non-stats line not matching the pattern)
in_section == 0 { next }

# Ignore the "---" separator line
/^  ---/ { next }

# Parse the stats lines
in_section == 1 && /assess  - / {
  n = $1 + 0   # numeric value

  # Remove leading spaces+number+spaces+assess+spaces+"- "
  # and capture the description.
  desc = $0
  sub(/^[[:space:]]*[0-9]+[[:space:]]*assess[[:space:]]*- /, "", desc)

  if (desc == "Assessments evaluated")
    printf("total=%d\n", n)
  else if (desc == "Resolved at compile time (static-true)")
    printf("static_true=%d\n", n)
  else if (desc == "Proven false at compile time (static-false)")
    printf("static_false=%d\n", n)
  else if (desc == "Runtime assertions generated")
    printf("runtime_total=%d\n", n)
  else if (desc == "Runtime assertions (entry cost)")
    printf("runtime_entry=%d\n", n)
  else if (desc == "Runtime assertions (low cost)")
    printf("runtime_low=%d\n", n)
  else if (desc == "Runtime assertions (medium cost)")
    printf("runtime_medium=%d\n", n)
  else if (desc == "Runtime assertions (high cost)")
    printf("runtime_high=%d\n", n)
  else if (desc == "Runtime assertions enabled")
    printf("runtime_enabled=%d\n", n)
  else if (desc == "Runtime assertions disabled by cost filter")
    printf("runtime_disabled=%d\n", n)
  else if (desc == "Assessments (unclassified)")
    printf("unclassified_total=%d\n", n)
  else if (desc == "Assessments (shape-compatibility)")
    printf("shape_compat_total=%d\n", n)
  else if (desc == "Assessments (element-access)")
    printf("elem_access_total=%d\n", n)
  else if (desc == "Assessments (loop-bound)")
    printf("loop_bound_total=%d\n", n)
  else if (desc == "Assessments (hw-constraint)")
    printf("hw_constraint_total=%d\n", n)
  else if (desc == "Runtime assertions (unclassified)")
    printf("unclassified_runtime=%d\n", n)
  else if (desc == "Runtime assertions (shape-compatibility)")
    printf("shape_compat_runtime=%d\n", n)
  else if (desc == "Runtime assertions (element-access)")
    printf("elem_access_runtime=%d\n", n)
  else if (desc == "Runtime assertions (loop-bound)")
    printf("loop_bound_runtime=%d\n", n)
  else if (desc == "Runtime assertions (hw-constraint)")
    printf("hw_constraint_runtime=%d\n", n)

  next
}

# If we hit a line that doesn't match the assess pattern and
# isn't a separator, we've left the section.
in_section == 1 && /./ && !/assess  - / {
  in_section = 0
}
