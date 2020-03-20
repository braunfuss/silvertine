#
unsetenv HYPOSAT_DATA
setenv HYPOSAT_DATA ../data

\cp ./hyposat-in hypomod-in
#\cp hyposat-parameter.tele.mod hyposat-parameter

# change here to your hyposat bin
/home/asteinbe/src/hyposat.6_0c/bin/hyposat
