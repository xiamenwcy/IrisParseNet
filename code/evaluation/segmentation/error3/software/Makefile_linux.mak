CXX=g++

# Linux
#CXXFLAGS=-O3 -Wall -fmessage-length=0 
CXXFLAGS=-g -Wall
LINKFLAGS=-s -L/opt/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_contrib -lboost_filesystem -lboost_system -lboost_regex -lopencv_photo

ALLTARGETS=maskcmpprf manuseg

maskcmpprf: maskcmpprf.cpp
	$(CXX) -o maskcmpprf $(CXXFLAGS) maskcmpprf.cpp $(LINKFLAGS)
	
manuseg: manuseg.cpp
	$(CXX) -o manuseg $(CXXFLAGS) manuseg.cpp $(LINKFLAGS)

all: ${ALLTARGETS}
clean:
	rm ${ALLTARGETS}
