reset
set xlabel 'Q [1/{\305}]'
set ylabel 'Diff. CS [barns/sr/atom]'
set style line 1 lt 1ps 0 lc 1
x=0
y=0
i=-1
plot \
'i15-1-18935_tth_det2_0.subcan' u 1:((column(2)+0.0)+0.0) notitle w l ls 1
