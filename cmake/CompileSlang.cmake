include(CMakeParseArguments)

# slang_compile_spirv(
#   NAME <unique_name>
#   SLANGC <path-to-slangc>            # optional if slangc is in PATH
#   SOURCE <file.slang>
#   OUT_DIR <dir>
#   FLAGS <...common flags...>         # optional
#   ENTRIES                           # list of triples: <entry> <stage> <outfile>
#     raygenMain raygeneration raygen.spv
#     missMain   miss         miss.spv
#     chitMain   closesthit   chit.spv
#     aHitMain   anyhit       ahit.spv
# )
#
# Exports to caller (PARENT_SCOPE):
#   <NAME>_SPV_FILES   : full paths to generated .spv files
#   <NAME>_SPV_TARGET  : custom target name (compile_shaders_<NAME>)
#
function(slang_compile_spirv)
  set(options)
  set(oneValueArgs NAME SLANGC SOURCE OUT_DIR)
  set(multiValueArgs FLAGS ENTRIES)
  cmake_parse_arguments(SLANG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT SLANG_NAME)
    message(FATAL_ERROR "slang_compile_spirv: NAME is required")
  endif()
  if(NOT SLANG_SOURCE)
    message(FATAL_ERROR "slang_compile_spirv: SOURCE is required")
  endif()
  if(NOT SLANG_OUT_DIR)
    message(FATAL_ERROR "slang_compile_spirv: OUT_DIR is required")
  endif()

  # Find slangc if not provided
  if(NOT SLANG_SLANGC)
    find_program(SLANG_SLANGC slangc)
    if(NOT SLANG_SLANGC)
      message(FATAL_ERROR "slang_compile_spirv: SLANGC not provided and 'slangc' not found in PATH")
    endif()
  endif()

  file(MAKE_DIRECTORY "${SLANG_OUT_DIR}")

  # ENTRIES must come as triples
  list(LENGTH SLANG_ENTRIES _len)
  math(EXPR _mod "${_len} % 3")
  if(NOT _mod EQUAL 0)
    message(FATAL_ERROR
      "slang_compile_spirv: ENTRIES must be triples: <entry> <stage> <outfile>. "
      "Got ${_len} items.")
  endif()

  set(spv_outputs "")
  set(cmds "")

  # Build command list per triple
  set(i 0)
  while(i LESS _len)
    list(GET SLANG_ENTRIES ${i} entry)
    math(EXPR i "${i}+1")
    list(GET SLANG_ENTRIES ${i} stage)
    math(EXPR i "${i}+1")
    list(GET SLANG_ENTRIES ${i} outname)
    math(EXPR i "${i}+1")

    set(outpath "${SLANG_OUT_DIR}/${outname}")
    list(APPEND spv_outputs "${outpath}")

    list(APPEND cmds
      COMMAND "${SLANG_SLANGC}" "${SLANG_SOURCE}" ${SLANG_FLAGS}
              -entry "${entry}" -stage "${stage}" -o "${outpath}"
    )
  endwhile()

  # One custom command generates all outputs
  add_custom_command(
    OUTPUT ${spv_outputs}
    ${cmds}
    DEPENDS "${SLANG_SOURCE}"
    COMMENT "Compiling Slang shaders (${SLANG_NAME}) to SPIR-V"
    VERBATIM
  )

  # Custom target you can depend on
  set(shader_target "compile_shaders_${SLANG_NAME}")
  add_custom_target(${shader_target} DEPENDS ${spv_outputs})

  # Export
  set(${SLANG_NAME}_SPV_FILES  ${spv_outputs} PARENT_SCOPE)
  set(${SLANG_NAME}_SPV_TARGET ${shader_target} PARENT_SCOPE)
endfunction()