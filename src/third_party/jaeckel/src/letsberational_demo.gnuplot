#!gnuplot

load "letsberational.gnuplot"

max(a,b) = a < b ? b : a
min(a,b) = a > b ? b : a

call_price = Black(1,1,1,1,1); sigma = ImpliedBlackVolatility(call_price,1,1,1,1); print "\tImpliedBlackVolatility(Black(1,1,1,1,1),1,1,1,1) = ",sigma,"\n"

x=-32

set term wxt persist size 960,800 linewidth 1.5 enhanced;
bind Close 'exit gnuplot'; bind Escape 'exit gnuplot'; # Only has an effect on some versions of gnuplot.

set multiplot layout 2,1

set key left Left reverse; set samples 2049; set grid

F=1; K=F*exp(-x); T=1; θ=1; Bmax = θ < 0 ? K : F

set title sprintf("x = %g, θ = %g",x,θ); set logscale y2; set y2tics; set xlabel 'σ'

σ_max = ImpliedBlackVolatility(Bmax*(1-DBL_EPSILON),F,K,T,θ)
σ_min = max(sqrt(DBL_MIN),ImpliedBlackVolatility(sqrt(DBL_MIN)*Bmax,F,K,T,θ))
plot [σ=σ_min:σ_max] Black(F,K,σ,T,θ) axis x1y1, ImpliedVolatilityAttainableAccuracy(x,σ*sqrt(T),θ) axis x1y2 t 'attainable relative accuracy [right ordinate]'

σ=sqrt(abs(2.0*x)); undefine x;
set title sprintf("σ = %g, θ = %g",σ,θ); unset ytics; set grid y2tics; set xlabel 'x'

restrict(y) = min(1.0,max(DBL_EPSILON/2.0,y))

x_min=-60; x_max=40
plot [x=x_min:x_max] \
  restrict(abs(ImpliedBlackVolatility(Black(F,F*exp(-x),σ,T,θ),F,F*exp(-x),T,θ)/σ-1)) axis x1y2 t 'attained relative accuracy', \
  restrict(ImpliedVolatilityAttainableAccuracy(x,σ*sqrt(T),θ)) axis x1y2 t 'attainable relative accuracy'

unset multiplot

#print "Hit Enter to end program"; pause -1

