libName_demo = AutoInfer
classFor_demo = Eason

srcs_lib_demo = srcs/demoClass/$(classFor_demo).cpp
objs_lib_demo = $(patsubst %.cpp,build/%.o,$(srcs_lib_demo))
lib_demo_cpp = libs/lib$(libName_demo)_cpp.so

src_main_demo = main_autoInfer.cpp
obj_main_demo = $(patsubst %.cpp,build/%.o,$(src_main_demo))

elf_main_demo = elf_$(subst main_,,$(subst .cpp,,src_main_demo))

default : $(elf_main_demo)

$(elf_main_demo) : $(obj_main_demo) $(lib_demo_cpp)
	@g++ -o $@ $< -L$(dir $(lib_demo_cpp)) -l$(subst .so,,$(subst libs/lib,,$(lib_demo_cpp)))
	@echo Build DNN sample : $@
	@LD_LIBRARY_PATH=libs ./$(elf_main_demo)

$(lib_demo_cpp) : $(objs_lib_demo)
	@g++ -fPIC -shared -Wl,-soname,$(notdir $(@)) -o $@ $^
	@echo generate $@

$(objs_lib_demo) : $(srcs_lib_demo)
	@mkdir -p $(dir $@)
	@g++ -o $@ -c $< -fPIC -I./include/demoClass
	@echo g++ $<

build/%.o : %.cpp
	@mkdir -p $(dir $@)
	@g++ -o $@ -c $< -I./include/demoClass
	@echo g++ $<

clean:
	@-rm -rf $(elf_main_demo) $(lib_demo_cpp) $(obj_main_demo) $(objs_lib_demo)
