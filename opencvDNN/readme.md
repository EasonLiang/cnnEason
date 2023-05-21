instruction for refactored DNN using opencv

tested on platforms :
	platform					depends
	======================================================
	<1/3> archlinux 	:: opencv 			, gcc v13
	<2/3> ubuntu22.04 	:: libopencv-dev	, gcc v11

<method 1> install dnnNet-3.2.4-ubuntu22.04.deb on ubuntu22.04 or dnnnet-3.2.4-1-x86_64.pkg.tar.zst on archlinux
	run command :
		$ elf_test

<method 2> compile and run inside directory bin
compile :
	make

test [ select corresponding file on your platform ]:
	$ cd bin
then [on ArchLinux]:
	./elf_test_Arch_Linux
or [on Ubuntu22.04]:
	./elf_test_Ubuntu_22_04_2
