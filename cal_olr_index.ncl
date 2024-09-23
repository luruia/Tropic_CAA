begin

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

yyyymm_s = 197901
yyyymm_e = 202112

;;;>-----------------------------------------------------------------------------------------<

f = addfile("./olr.mon.mean.nc", "r")
time = cd_calendar(f->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
pre = f->olr(ind_s:ind_e,:,:)

;;;>-----------------------------------------------------------------------------------------<

dims = dimsizes(pre)
n_year = dims(0)/12

pre_1 = reshape(pre, (/n_year, 12, dims(1), dims(2)/))
pre_jjas = dim_avg_n_Wrap(pre_1(:,5:8,:,:), 1)
copy_VarCoords(pre(0,:,:), pre_jjas(0,:,:))
pre_jjas!0 = "time"
pre_jjas&time = ispan(1979, 2021, 1)
printVarSummary(pre_jjas)

pre_jjas = dtrend_msg_n(pre_jjas&time, pre_jjas, False, False, 0)

;;;>-----------------------------------------------------------------------------------------<

pre_std = dim_stddev_n_Wrap(pre_jjas, 0)
pre_clm = dim_avg_n_Wrap(pre_jjas, 0)
copy_VarCoords(pre(0,:,:), pre_std)
copy_VarCoords(pre(0,:,:), pre_clm)
delete(pre)
delete(pre_1)

;;;>-----------------------------------------------------------------------------------------<

;;;   寻找大值点
temp = pre_std({lat|-15:25},{lon|60:160})
lat_temp = temp&lat
lon_temp = temp&lon
nmax = local_max(temp, False, 0)
x = lon_temp(nmax@xi)
y = lat_temp(nmax@yi)
print(x)
print(y)
; k = 0 ;;  dtrend
k = 0 ;;  trend

print(pre_std({y(k)},{x(k)}))
pc_max = pre_jjas(:, {y(k)},{x(k)})
corr_var = escorc(pc_max, pre_jjas({lat|:},{lon|:},{time|:}))
copy_VarCoords(pre_jjas(0,:,:), corr_var)
; printVarSummary(corr_var)
prob = rtest(corr_var, n_year, 0)
copy_VarCoords(pre_jjas(0,:,:), prob)

;;;>-----------------------------------------------------------------------------------------<

gres = True
gres@gsLineColor = "blue"
gres@gsLineThicknessF = 5
gres@gsLineDashPattern = 2


lonw = 80
lonc = 180
lone = 260
lats = -15
latn = 15
x1 = (/lonw,lonc,lone,lone,lonc,lonw,lonw/)
y1 = (/lats,lats,lats,latn,latn,latn,lats/)

;;;>-----------------------------------------------------------------------------------------<

olr_wgt = wgt_areaave2(pre_jjas(:,{lats:latn},{lonw:lone}), corr_var({lats:latn},{lonw:lone}), 0)
olr_wgt = olr_wgt * (-1.)
olr_wgt = dim_standardize(olr_wgt, 1)


corr_pre = escorc(olr_wgt, pre_jjas({lat|:},{lon|:},{time|:}))
copy_VarCoords(pre_jjas(0,:,:), corr_pre)
; printVarSummary(corr_var)
prob_pre = rtest(corr_pre, n_year, 0)
copy_VarCoords(pre_jjas(0,:,:), prob_pre)

;;;>-----------------------------------------------------------------------------------------<

system("rm -rf ./olr_index.nc")
ff = addfile("./olr_index.nc", "c")
ff->pc = olr_wgt

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

;;;   绘图
res_index = True
res_index@gsnFrame = False
res_index@gsnDraw = False
res_index@gsnLeftString = ""
res_index@gsnRightString = ""
res_index@gsnCenterString = ""
res_index@gsnAddCyclic = True
res_index@tmBorderThicknessF = 4
res_index@gsnStringFont = 26
res_index@tmXBLabelFont = 7
res_index@tmYLLabelFont = 7
res_index@gsnLeftString = "OLR Index"
res_index@gsnLeftStringFontHeightF = 22

res_index@vpWidthF = 0.7
res_index@vpHeightF = 0.35
res_index@gsnYRefLine = 0 

res_index@trYMaxF = 3.0
res_index@trYMinF = -3.0
res_index@trXMaxF = 2022
res_index@trXMinF = 1978

res_index@tmXTMinorOn = False
res_index@tmYRMinorOn = False
res_index@tmXBMinorOn = False
res_index@tmYLMinorOn = False
res_index@tmXBMajorLengthF= -0.005
res_index@tmYLMajorLengthF= -0.005
res_index@tmYLLabelDeltaF = -0.7
res_index@tmXBLabelDeltaF = -0.7
res_index@tmXBLabelFontHeightF = 0.016
res_index@tmYLLabelFontHeightF = 0.016

res_index@tmXMajorGrid = True
res_index@tmXMajorGridThicknessF = 0.3
res_index@tmXMajorGridLineDashPattern = 2
res_index@tmXMajorGridLineColor = "grey"
res_index@tmYMajorGrid = True
res_index@tmYMajorGridThicknessF = 0.3
res_index@tmYMajorGridLineDashPattern = 2
res_index@tmYMajorGridLineColor = "grey"
res_index@tmGridDrawOrder = "PreDraw"

res_index@tmXBMode = "Explicit"
res_index@tmXBValues = ispan(1979, 2021, 5)
res_index@tmXBLabels = ispan(1979, 2021, 5)
res_index@xyMarkLineMode = "MarkLines"
res_index@xyMarker = 1
res_index@xyMarkerSizeF = 30
res_index@xyLineThicknessF = 2

wks11 = gsn_open_wks("x11", "./olr_wgt_index")
plot11 = gsn_csm_xy(wks11, ispan(1979, 2021, 1), olr_wgt, res_index)

draw(plot11)
frame(wks11)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

end