#!gnuplot

import Revision(dummy)					from "letsberational_in_gnuplot:revision"
import Bitness(dummy)					from "letsberational_in_gnuplot:bitness"
import CPUName(dummy)					from "letsberational_in_gnuplot:cpuname"
import dllname(dummy)					from "letsberational_in_gnuplot"
import builddate(dummy)					from "letsberational_in_gnuplot"
import dlldirectory(dummy)				from "letsberational_in_gnuplot"
import compilerversion(dummy)			from "letsberational_in_gnuplot"
import buildconfiguration(dummy)		from "letsberational_in_gnuplot"

import DblEpsilon(dummy)				from "letsberational_in_gnuplot:dblepsilon"
DBL_EPSILON = DblEpsilon(0)
import DblMin(dummy)					from "letsberational_in_gnuplot:dblmin"
DBL_MIN = DblMin(0)
import DblMax(dummy)					from "letsberational_in_gnuplot:dblmax"
DBL_MAX = DblMax(0)

import Black(F,K,sigma,T,q)							from "letsberational_in_gnuplot:black"
import ImpliedBlackVolatility(value,F,K,T,q)		from "letsberational_in_gnuplot:impliedblackvolatility"
import ImpliedVolatilityAttainableAccuracy(x,s,q)	from "letsberational_in_gnuplot:impliedvolatilityattainableaccuracy"

import Normalised_Black(x,s,q)						from "letsberational_in_gnuplot:normalisedblack"
import lets_be_rational(beta,qx,n)					from "letsberational_in_gnuplot:letsberational"

print "\nLoaded dynamic load library ",dllname(0)," revision ",Revision(0)," [",Bitness(0)," bit ",buildconfiguration(0)," running on ",CPUName(0),"] compiled with ",compilerversion(0)," on ",builddate(0)," from ",dlldirectory(0)," .\n"
