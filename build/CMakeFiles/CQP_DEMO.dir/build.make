# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake3

# The command to remove a file.
RM = /usr/bin/cmake3 -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tang_wenqi/computational_quantum_physics

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tang_wenqi/computational_quantum_physics/build

# Include any dependencies generated for this target.
include CMakeFiles/CQP_DEMO.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CQP_DEMO.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CQP_DEMO.dir/flags.make

CMakeFiles/CQP_DEMO.dir/src/main.cpp.o: CMakeFiles/CQP_DEMO.dir/flags.make
CMakeFiles/CQP_DEMO.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tang_wenqi/computational_quantum_physics/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CQP_DEMO.dir/src/main.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CQP_DEMO.dir/src/main.cpp.o -c /home/tang_wenqi/computational_quantum_physics/src/main.cpp

CMakeFiles/CQP_DEMO.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CQP_DEMO.dir/src/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tang_wenqi/computational_quantum_physics/src/main.cpp > CMakeFiles/CQP_DEMO.dir/src/main.cpp.i

CMakeFiles/CQP_DEMO.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CQP_DEMO.dir/src/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tang_wenqi/computational_quantum_physics/src/main.cpp -o CMakeFiles/CQP_DEMO.dir/src/main.cpp.s

CMakeFiles/CQP_DEMO.dir/src/cqp_time_test.cpp.o: CMakeFiles/CQP_DEMO.dir/flags.make
CMakeFiles/CQP_DEMO.dir/src/cqp_time_test.cpp.o: ../src/cqp_time_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tang_wenqi/computational_quantum_physics/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/CQP_DEMO.dir/src/cqp_time_test.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CQP_DEMO.dir/src/cqp_time_test.cpp.o -c /home/tang_wenqi/computational_quantum_physics/src/cqp_time_test.cpp

CMakeFiles/CQP_DEMO.dir/src/cqp_time_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CQP_DEMO.dir/src/cqp_time_test.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tang_wenqi/computational_quantum_physics/src/cqp_time_test.cpp > CMakeFiles/CQP_DEMO.dir/src/cqp_time_test.cpp.i

CMakeFiles/CQP_DEMO.dir/src/cqp_time_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CQP_DEMO.dir/src/cqp_time_test.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tang_wenqi/computational_quantum_physics/src/cqp_time_test.cpp -o CMakeFiles/CQP_DEMO.dir/src/cqp_time_test.cpp.s

# Object files for target CQP_DEMO
CQP_DEMO_OBJECTS = \
"CMakeFiles/CQP_DEMO.dir/src/main.cpp.o" \
"CMakeFiles/CQP_DEMO.dir/src/cqp_time_test.cpp.o"

# External object files for target CQP_DEMO
CQP_DEMO_EXTERNAL_OBJECTS =

CQP_DEMO: CMakeFiles/CQP_DEMO.dir/src/main.cpp.o
CQP_DEMO: CMakeFiles/CQP_DEMO.dir/src/cqp_time_test.cpp.o
CQP_DEMO: CMakeFiles/CQP_DEMO.dir/build.make
CQP_DEMO: /opt/rh/devtoolset-11/root/usr/lib/gcc/x86_64-redhat-linux/11/libgomp.so
CQP_DEMO: /lib64/libpthread.so
CQP_DEMO: CMakeFiles/CQP_DEMO.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tang_wenqi/computational_quantum_physics/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable CQP_DEMO"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CQP_DEMO.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CQP_DEMO.dir/build: CQP_DEMO

.PHONY : CMakeFiles/CQP_DEMO.dir/build

CMakeFiles/CQP_DEMO.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CQP_DEMO.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CQP_DEMO.dir/clean

CMakeFiles/CQP_DEMO.dir/depend:
	cd /home/tang_wenqi/computational_quantum_physics/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tang_wenqi/computational_quantum_physics /home/tang_wenqi/computational_quantum_physics /home/tang_wenqi/computational_quantum_physics/build /home/tang_wenqi/computational_quantum_physics/build /home/tang_wenqi/computational_quantum_physics/build/CMakeFiles/CQP_DEMO.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CQP_DEMO.dir/depend

