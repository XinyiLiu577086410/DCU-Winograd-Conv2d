# final
hipprof --hip-trace --pmc ./conv2dfp16demo 16 128 64 64 27 3 3 1 1 1 1
hipprof --hip-trace --pmc ./conv2dfp16demo 16 256 32 32 256 3 3 1 1 1 1
hipprof --hip-trace --pmc ./conv2dfp16demo 16 64 128 128 64 3 3 1 1 1 1
hipprof --hip-trace --pmc ./conv2dfp16demo 2 1920 32 32 640 3 3 1 1 1 1
hipprof --hip-trace --pmc ./conv2dfp16demo 2 640  64 64 640 3 3 1 1 1 1
hipprof --hip-trace --pmc ./conv2dfp16demo 2 320  64 64 4 3 3 1 1 1 1

