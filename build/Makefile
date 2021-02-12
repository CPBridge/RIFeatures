# Compiler
CXX:=g++

# Flags
CXX_FLAGS_RELEASE:=-Ofast -march=native -DNDEBUG -fopenmp -Wall -Wextra -std=c++11
CXX_FLAGS_DEBUG:=-ggdb -fopenmp -Wall -Wextra -std=c++11

# Source files
HEADERS=../include/RIFeatures/RIFeatExtractor.hpp ../include/RIFeatures/RIFeatExtractor.tpp ../include/RIFeatures/struve.hpp
SOURCE=../src/RIFeatExtractor.cpp ../src/struve.cpp $(HEADERS)

# Installation directory
LIB_INSTALL_DIR:=/usr/local/lib/
HEADER_INSTALL_DIR:=/usr/local/include/RIFeatures/

all: librifeatures.so

debug: librifeaturesd.so

# Release version
librifeatures.so: $(SOURCE)
	$(CXX) $(CXX_FLAGS_RELEASE) -I ../include/ -fPIC -shared -o $@ ../src/RIFeatExtractor.cpp ../src/struve.cpp

# Debug version
librifeaturesd.so: $(SOURCE)
	$(CXX) $(CXX_FLAGS_DEBUG) -I ../include/ -fPIC -shared -o $@ ../src/RIFeatExtractor.cpp ../src/struve.cpp

install:
	cp librifeatures.so $(LIB_INSTALL_DIR)
	mkdir $(HEADER_INSTALL_DIR)
	cp $(HEADERS) $(HEADER_INSTALL_DIR)

install-debug:
	cp librifeaturesd.so $(LIB_INSTALL_DIR)
	mkdir $(HEADER_INSTALL_DIR)
	cp $(HEADERS) $(HEADER_INSTALL_DIR)

clean:
	rm -f *.so example

uninstall:
	rm -f $(LIB_INSTALL_DIR)librifeatures.so $(LIB_INSTALL_DIR)librifeaturesd.so
	rm -f $(HEADER_INSTALL_DIR)*
	rm -rf $(HEADER_INSTALL_DIR)

example:
	$(CXX) $(CXX_FLAGS_RELEASE) -I ../include/ -o example ../example/rotinv_test.cpp `pkg-config --libs opencv4` -L . -l rifeatures
