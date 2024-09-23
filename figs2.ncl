
begin

;>---------------------------------------------------------------------------------------<

yyyymm_s = 197901
yyyymm_e = 202112

;>---------------------------------------------------------------------------------------<
;;                  读取数据 
;>---------------------------------------------------------------------------------------<

f = addfile("./olr.mon.mean.nc", "r")
time = cd_calendar(f->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
pre = f->olr(ind_s:ind_e,:,:)
pre = lonFlip(pre)

dims = dimsizes(pre)
n_year = dims(0)/12

pre_1 = reshape(pre, (/n_year, 12, dims(1), dims(2)/))
pre_jjas = dim_avg_n_Wrap(pre_1(:,5:8,:,:), 1)
copy_VarCoords(pre(0,:,:), pre_jjas(0,:,:))
pre_jjas!0 = "time"
pre_jjas&time = ispan(1979, 2021, 1)
printVarSummary(pre_jjas)
dims1 = dimsizes(pre_jjas)

;>---------------------------------------------------------------------------------------<
;;                  计算权重 
;>---------------------------------------------------------------------------------------<

w  = sqrt(cos(0.01745329 * pre_jjas&lat) )
wp = pre_jjas * conform(pre_jjas, w, 1)
copy_VarCoords(pre_jjas, wp)
wp = lonFlip(wp)
printVarSummary(wp)

;>---------------------------------------------------------------------------------------<
;;                  EOF分解
;>---------------------------------------------------------------------------------------<

minlat = -20
maxlat = 20
minlon = 60
maxlon = 290

x = wp({lat|minlat:maxlat},{lon|minlon:maxlon},{time|:})
neof = 10
eof = eofunc_Wrap(x, neof, False)
eof = -eof
tzh = eof@pcvar
lam = tzh * sqrt(2. / dims1(0))       ;计算解释方差误差范围

;>---------------------------------------------------------------------------------------<
;;                  north检验
;>---------------------------------------------------------------------------------------<

ntim = dims1(0)               
sig_pcv = eofunc_north(eof@pcvar, ntim, True)

;>---------------------------------------------------------------------------------------<
;;                 计算时间序列并标准化
;>---------------------------------------------------------------------------------------<

eof_ts = eofunc_ts_Wrap (x, eof, False)
eof_ts_stddev = dim_stddev(eof_ts)       ;计算时间系数标准差
eof_ts = dim_standardize_n(eof_ts,0,1)

ts_dev = dim_stddev_n(eof_ts, 1)

;>---------------------------------------------------------------------------------------<
;                 将标准差与空间模态相乘
;>---------------------------------------------------------------------------------------<

do i = 0, 9
    eof(i,:,:) = (/ eof(i,:,:) /) * (/ eof_ts_stddev(i) /)  
end do

;>---------------------------------------------------------------------------------------<
;;                      绘图
;>---------------------------------------------------------------------------------------<

wks1 = gsn_open_wks("pdf", "./figs2")

res = True
res@gsnDraw = False
res@gsnFrame = False
res@gsnCenterString = ""
res@gsnRightString = ""
res@gsnLeftString = ""
res@tmBorderThicknessF = 2
res@gsnLeftStringOrthogonalPosF = 0
res@gsnLeftStringFontHeightF = 0.02

res@tmXBMinorLengthF = 0
res@tmYLMinorLengthF = 0
res@tmXBMajorLengthF = 0.005
res@tmYLMajorLengthF = 0.005
res@tmXBLabelDeltaF = -0.5
res@tmYLLabelDeltaF = -0.5
res@tmXBLabelFontThicknessF = 1.5
res@tmYLLabelFontThicknessF = 1.5
res@tmXBLabelFontHeightF = 0.015
res@tmYLLabelFontHeightF = 0.015
res@tmXBMajorThicknessF = 2
res@tmYLMajorThicknessF = 2
res@tmXTOn = False
res@tmYROn = False

rxy = res
res_sig = res
res_clm = res
resv = res

res@gsnAddCyclic = False

res@mpCenterLonF = 180
res@mpMinLatF = -35
res@mpMaxLatF = 35
res@mpFillOn = False
res@mpShapeMode = "FreeAspect"
res@tmXBTickSpacingF = 45
res@tmYLTickSpacingF = 20

res@cnFillPalette = "BlWhRe"
res@cnFillOn = True
res@cnLinesOn = False
res@cnLineLabelsOn = False
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnLevels = (/-1.,-0.9,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0/)*10.
res@cnFillColors = (/30,35,38,41,44,-1,60,65,70,75,80/)
; res@cnFillOpacityF = 0.6
res@lbLabelBarOn = True
res@pmLabelBarOrthogonalPosF = 0.2
res@lbBoxEndCapStyle = "TriangleBothEnds"


res@vpWidthF = 0.8
res@vpHeightF = 0.2
res@vpXF = 0.1
res@vpYF = 0.83
res@pmLabelBarWidthF = 0.6
res@pmLabelBarHeightF = 0.05
res@pmLabelBarOrthogonalPosF = 0.15
; res@pmLabelBarParallelPosF = 1.1
res@gsnLeftString = "(a) EOF1 ("+ decimalPlaces(eof@pcvar(0), 1, True) +"%)"
plot1 = gsn_csm_contour_map(wks1, eof(0,:,:), res)

draw(plot1)

delete([/res, res_clm, res_sig/])

;>---------------------------------------------------------------------------------------<


rxy@vpXF = 0.1
rxy@vpYF = 0.52
rxy@vpWidthF = 0.55
rxy@vpHeightF = 0.2

xx = ispan(yyyymm_s/100, yyyymm_e/100, 1)

rxy@trXMinF = yyyymm_s/100-1
rxy@trXMaxF = yyyymm_e/100+1
rxy@trYMinF = -3.5
rxy@trYMaxF = 3.5
rxy@gsnYRefLine = 0     ;设定参考值

rxy@tmYLMode = "Explicit"
rxy@tmYLValues = (/-3,-2,-1,0,1,2,3/)
rxy@tmYLLabels = (/-3,-2,-1,0,1,2,3/)
rxy@tmXBMode = "Explicit"
rxy@tmXBValues = ispan(yyyymm_s/100, yyyymm_e/100, 5)
rxy@tmXBLabels = ispan(yyyymm_s/100, yyyymm_e/100, 5)

rxy@xyLineColor = "blue"
rxy@xyLineThicknessF = 4
rxy@pmLegendHeightF = 0.08
rxy@pmLegendWidthF = 0.1
rxy@pmLegendOrthogonalPosF = -1.4
rxy@pmLegendParallelPosF = 0.2
rxy@lgBoxMinorExtentF = 0.4
rxy@lgLabelFontHeightF = 0.01
rxy@lgLabelFontHeightF = 0.015
rxy@lgPerimOn = False
rxy@gsnLeftString = "(b) PC1"
rxy@gsnLeftStringOrthogonalPosF = 0

plotxx = gsn_csm_xy(wks1, xx, eof_ts(0,:), rxy)
draw(plotxx)

;>---------------------------------------------------------------------------------------<

x_variance_all = eof@pcvar


resv@trXMaxF = 5
resv@trXMinF = -1
resv@trYMaxF = 41
resv@trYMinF = 0
resv@tmXBMinorLengthF = 0
resv@tmYLMinorLengthF = 0
resv@tmXBMajorLengthF = 0.005
resv@tmYLMajorLengthF = 0.005
resv@tmXBLabelDeltaF = -0.5
resv@tmYLLabelDeltaF = -0.5
resv@tmXBLabelFontThicknessF = 1.5
resv@tmYLLabelFontThicknessF = 1.5
resv@tmXBLabelFontHeightF = 0.015
resv@tmYLLabelFontHeightF = 0.015
resv@tmXBMajorThicknessF = 2
resv@tmYLMajorThicknessF = 2
resv@tmXTOn = False
resv@tmYROn = False
resv@tmXBMode = "Explicit"
resv@tmXBValues = (/0,1,2,3,4,5,6,7,8,9/)
resv@tmXBLabels = (/"1","2","3","4","5","","","","",""/)

resv@vpXF = 0.7
resv@vpYF = 0.52
resv@vpWidthF = 0.2
resv@vpHeightF = 0.2

resv@gsnLeftString = "(c) Variance"

;;;; 添加error bar

plot3 = gsn_csm_y(wks1, x_variance_all(:4), resv)

polyres = True
polyres@gsMarkerIndex = 1 
polyres@gsMarkerSizeF = .02
polyres@gsLineColor = "blue"
polyres@gsLineThicknessF = 2.5

centers = new(dimsizes(x_variance_all),graphic)
l1dum = new(dimsizes(x_variance_all),graphic) 
l2dum = new(dimsizes(x_variance_all),graphic)
l3dum = new(dimsizes(x_variance_all),graphic)

; do t=0,dimsizes(x_variance_all)-1
do t=0,4
  centers(t)   = gsn_add_polymarker(wks1,plot3,t,x_variance_all(t),polyres)
  l1dum(t) = gsn_add_polyline(wks1,plot3,(/t,t/),(/x_variance_all(t)-lam(t),x_variance_all(t)+lam(t)/),polyres)
  l2dum(t) = gsn_add_polyline(wks1,plot3,(/t-.15,t+.15/),(/x_variance_all(t)+lam(t),x_variance_all(t)+lam(t)/),polyres)
  l3dum(t) = gsn_add_polyline(wks1,plot3,(/t-.15,t+.15/),(/x_variance_all(t)-lam(t),x_variance_all(t)-lam(t)/),polyres)
end do
;>---------------------------------------------------------------------------------------<
draw(plot3)
frame(wks1)

end