libName_demo = AutoInfer
classFor_demo = Eason

linuxDistStr = $(shell head -1 /etc/issue | awk '{printf("%s%s\n",$$1,$$2)}' | sed -e 's/\./_/g' )


srcs_lib_cpp = srcs/autoInferClass/$(classFor_demo).cpp
# section for so to be invoked by c or cpp
	objs_lib_cpp = $(patsubst %.cpp,build/%.o,$(srcs_lib_cpp))
lib_demo_cpp = libs/lib$(libName_demo)_$(linuxDistStr).so

# section for main for c or cpp
src_main_demo = main_autoInfer.cpp
	obj_main_demo = $(patsubst %.cpp,build/%.o,$(src_main_demo))

elf_main_demo = elf_$(subst main_,,$(subst .cpp,,$(src_main_demo)))_$(linuxDistStr)

# section for so to be invoked by python
swig_intf4python = include/demo_cpp2python.i
	python_module_name = $(shell head -1 $(swig_intf4python) | cut -d\  -f2 ).py
	srcs_wrap4python = build/wrap/$(subst .py,,$(python_module_name))_wrap.cxx
	objs_wrap4python = $(patsubst %.cxx,%.o,$(srcs_wrap4python))

	pyIncVer=$(shell realpath $(shell which python)|cut -d/ -f4)

libs_demo_cpp2py = libs/$(linuxDistStr)/_$(libName_demo)_cpp2python_$(linuxDistStr).so

default : $(elf_main_demo) $(libs_demo_cpp2py)

$(elf_main_demo) : $(obj_main_demo) $(lib_demo_cpp)
	@g++ -o $@ $< -L$(dir $(lib_demo_cpp)) -l$(subst .so,,$(subst libs/lib,,$(lib_demo_cpp)))
	@echo Build DNN sample : $@
	@echo -e "\t================================"
#	@LD_LIBRARY_PATH=libs ./$(elf_main_demo)

$(lib_demo_cpp) : $(objs_lib_cpp)
	@mkdir -p $(dir $@)
	@g++ -shared -Wl,-soname,$(notdir $(@)) -o $@ $^
	@echo generate $@

$(objs_lib_cpp) : $(srcs_lib_cpp)
	@mkdir -p $(dir $@)
	@g++ -o $@ -c $< -fPIC -std=c++11 -I./include/autoInferClass
	@echo g++ $<

build/%.o : %.cpp
	@mkdir -p $(dir $@)
	@g++ -o $@ -c $< -std=c++11 -I./include/autoInferClass
	@echo g++ $<

$(libs_demo_cpp2py): $(objs_lib_cpp) $(objs_wrap4python)
	@mkdir -p $(dir $@)
	@g++ -shared -o $@ $^
	@if [ ! -f libs/$(linuxDistStr)/$(python_module_name) ];then cp -r $(dir $(srcs_wrap4python))/$(python_module_name) libs/$(linuxDistStr);fi
	@echo Generate so file to be invoked by python from CPP

$(objs_wrap4python) : $(srcs_wrap4python)
	@mkdir -p $(dir $@)
	@g++ -o $@ -c $< -fPIC -std=c++11 -I /usr/include/$(pyIncVer) -I include
	@echo build obj file for cpp\'s wrap file

$(srcs_wrap4python) : $(swig_intf4python)
	@mkdir -p $(dir $@)
	@sed -i 's@_ArchLinux@_$(linuxDistStr)@g' $<
	@swig -c++ -python -o $@ $<
	@sed -i '1i\__asm__(".symver realpath,realpath@GLIBC_2.2.5");'	$@
	@echo generate wrap file for cpp

clall : clean
	@-rm -rf libs/$(linuxDistStr)/$(python_module_name) $(libs_demo_cpp2py)

clean: cleandemo
	@-rm -rf $(elf_main_demo) $(objs_lib_cpp) $(obj_main_demo) $(lib_demo_cpp)

cleandemo:
	@-rm -rf $(srcs_wrap4python) $(objs_wrap4python) $(dir $(srcs_wrap4python))/$(python_module_name)

#clean:
#	@-rm -rf $(elf_main_demo) $(lib_demo_cpp) $(obj_main_demo) $(objs_lib_demo)
