build_dir = build

verStr=$(shell head -1 /etc/issue | awk '{print $$1,$$2}'|sed -e 's/ \|\./_/g')

INC = -I./include -I /usr/include/opencv4
FLAGS_CPP = -std=c++11
LIB_SRCS = lib/libeasonNet.so

SRCS_EXPL = $(wildcard samples/*.cpp)
ELFS = $(patsubst %.cpp,bin/elf_%_$(verStr),$(notdir $(SRCS_EXPL)))

default : $(ELFS)

bin/elf_%_$(verStr) : build/samples/%.o $(LIB_SRCS)
	@echo $@
	@mkdir -p $(dir $@)
	@echo -e "\t Generating $@"
	@g++ -o $@ $^ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml

$(build_dir)/%.o : %.cpp
	@mkdir -p $(dir $@)
	@g++ -c -o $@ $< $(INC) $(FLAGS_CPP)
	@echo CPP $<

.phony : clean

clean :
	@-rm -rf $(OBJS_SRCS_LIB) $(ELFS) build/samples/*.o

.SECONDARY :
