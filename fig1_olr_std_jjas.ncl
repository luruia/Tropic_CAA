begin

;;;>-----------------------------------------------------------------------------------------<

yyyymm_s = 197901
yyyymm_e = 202112

; f = addfile("/mnt/d/data/1-initial/reanalysis/noaa/interploted_olr/olr-monthly_v02r07_197901_202303.nc", "r")
f = addfile("/mnt/d/data/1-initial/reanalysis/noaa/interploted_olr/olr.mon.mean.nc", "r")
time = cd_calendar(f->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
pre = f->olr(ind_s:ind_e,:,:)

dims = dimsizes(pre)
n_year = dims(0)/12

pre_1 = reshape(pre, (/n_year, 12, dims(1), dims(2)/))
pre_jjas = dim_avg_n_Wrap(pre_1(:,5:8,:,:), 1)
copy_VarCoords(pre(0,:,:), pre_jjas(0,:,:))
pre_jjas!0 = "time"
pre_jjas&time = ispan(1979, 2021, 1)
printVarSummary(pre_jjas)

; pre_jjas = dtrend_msg_n(pre_jjas&time, pre_jjas, False, False, 0)

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


olr_wgt = wgt_areaave2(pre_jjas(:,{lats:latn},{lonw:lone}), corr_var({lats:latn},{lonw:lone}), 0)
olr_wgt = olr_wgt * (-1.)
olr_wgt = dim_standardize(olr_wgt, 1)


corr_pre = escorc(olr_wgt, pre_jjas({lat|:},{lon|:},{time|:}))
copy_VarCoords(pre_jjas(0,:,:), corr_pre)
; printVarSummary(corr_var)
prob_pre = rtest(corr_pre, n_year, 0)
copy_VarCoords(pre_jjas(0,:,:), prob_pre)


; fout = addfile("./diagram_olr.nc", "c")
; fout->olr = corr_pre

; system("rm -rf ./olr_index_trend.nc")
; ff = addfile("./olr_index_trend.nc", "c")
; ff->pc = olr_wgt


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
; res_index@gsnAboveYRefLineColor = "red"
; res_index@gsnBelowYRefLineColor = "blue"

wks11 = gsn_open_wks("x11", "./olr_wgt_index")
plot11 = gsn_csm_xy(wks11, ispan(1979, 2021, 1), olr_wgt, res_index)

draw(plot11)
frame(wks11)



res = True
res@gsnFrame = False
res@gsnDraw = False
res@gsnLeftString = ""
res@gsnRightString = ""
res@gsnCenterString = ""
res@tmBorderThicknessF = 3
res@gsnStringFont = 22
; res@tmXBLabelFont = 7
; res@tmYLLabelFont = 7

resm = res
res_sig = res

res@vpWidthF = 0.8
res@vpHeightF = 0.42
res@mpShapeMode = "FreeAspect"
res@mpMinLatF = -40
res@mpMaxLatF = 40
; res@mpMinLonF = 45
; res@mpMaxLonF = 270
res@mpCenterLonF = 180
res@mpFillOn = False
; res@mpLandFillColor = "white"
; res@mpFillDrawOrder = "PostDraw"
; res@mpOutlineDrawOrder = "PreDraw"
res@mpGeophysicalLineColor = "gray40"
res@mpGeophysicalLineThicknessF = 2.0
; res@gsnLeftStringOrthogonalPosF = 0.01
res@gsnLeftStringFontHeightF = 25
; res@gsnRightStringOrthogonalPosF = 0.01
res@gsnRightStringFontHeightF = 25

res@tmXBMinorLengthF = 0
res@tmYLMinorLengthF = 0
res@tmXBMajorLengthF = 0.005
res@tmYLMajorLengthF = 0.005
res@tmXBLabelDeltaF = -0.5
res@tmYLLabelDeltaF = -0.5
res@tmXBLabelFontThicknessF = 1.5
res@tmYLLabelFontThicknessF = 1.5
res@tmXBLabelFontHeightF = 0.02
res@tmYLLabelFontHeightF = 0.02
res@tmXBTickSpacingF = 45
res@tmYLTickSpacingF = 20
res@tmXBMajorThicknessF = 2.5
res@tmYLMajorThicknessF = 2.5
res@tmXTOn = False
res@tmYROn = False

rescor = res

res@cnLinesOn = False
res@cnLineLabelsOn = False
res@cnInfoLabelOn = False
res@cnFillOn = True
res@cnFillPalette = "MPL_PuOr"
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnLevels := (/6,9,12,15,18,21/)
res@cnFillColors = (/-1,50,40,30,20,10,5/)

res@lbLabelBarOn = True
res@lbOrientation = "Vertical"
res@pmLabelBarWidthF = 0.04
res@lbLabelFontHeightF = 0.015
res@lbBoxEndCapStyle = "TriangleBothEnds"

resm@cnLinesOn = True
resm@cnLineLabelsOn = True
resm@cnInfoLabelOn = False
resm@cnFillOn = False
resm@cnLineLabelFont = 7
resm@cnLineLabelFontHeightF = 0.01
resm@cnLineLabelBackgroundColor = -1
resm@cnLineLabelAngleF = 0.0
resm@cnLineLabelDensityF = 1
resm@cnLineLabelInterval = 1
resm@cnLineLabelPerimOn = False
resm@cnLineLabelFormat = "0@;*.2f"
resm@cnLineDrawOrder = "PostDraw"
resm@cnLabelDrawOrder = "PostDraw"

resm@cnLineThicknessF = 2.5
resm@cnLevelSelectionMode = "ExplicitLevels"
resm@cnLevels = (/220,240/)
resm@gsnContourZeroLineThicknessF = 0.0

rescor@cnFillOn = True
rescor@cnLinesOn = False
rescor@cnLineLabelsOn = False
rescor@cnInfoLabelOn = False
rescor@cnLevelSelectionMode = "ExplicitLevels"
rescor@cnLevels = (/-0.8,-0.5,-0.3,0.3,0.5,0.8/)
rescor@cnLineDrawOrder = "PostDraw"
rescor@cnLineThicknessF = 2
rescor@cnFillOpacityF = 0.7
rescor@lbLabelBarOn = True
rescor@lbOrientation = "Vertical"
rescor@pmLabelBarWidthF = 0.04
rescor@lbLabelFontHeightF = 0.015
rescor@lbBoxEndCapStyle = "TriangleBothEnds"
; rescor@cnFillPalette = "BlWhRe"
rescor@cnFillPalette = "testcmap"
rescor@cnFillColors = (/72,84,94,-1,115,135,155/)

res_sig@cnFillOn = True
res_sig@lbLabelBarOn = False
res_sig@cnLinesOn = False
res_sig@cnInfoLabelOn = False
res_sig@cnLineLabelsOn = False
res_sig@cnMonoFillPattern = False
res_sig@cnLevelSelectionMode = "ExplicitLevels"
res_sig@cnLevels = (/0.1/)
res_sig@cnFillPatterns = (/8, -1/)
res_sig@cnFillColors = (/1, 0/)
res_sig@cnFillDotSizeF = 0
res_sig@cnFillDrawOrder = "PostDraw"

pmres = True
pmres@gsMarkerIndex = 12
pmres@gsMarkerSizeF = 0.02
pmres@gsMarkerThicknessF = 3.5
pmres@gsMarkerColor = "green"


wks = gsn_open_wks("pdf", "std_cor_olr_trend-4")

res@gsnLeftString = "(a) Std. of OLR in JJAS"
plot = gsn_csm_contour_map(wks, pre_std, res)
plot2 = gsn_csm_contour(wks, pre_clm, resm)
dum2 = gsn_add_polyline(wks, plot, x1, y1, gres)
; dum22 = gsn_add_polyline(wks, plot, x3, y3, gres)
; dum51 = gsn_add_polyline(wks, plot, x2, y2, gres)
dum01 = gsn_add_polymarker(wks, plot2, x(k), y(k), pmres)
overlay(plot, plot2)

; draw(plot)
; frame(wks)


rescor@gsnLeftString = "(b) Point Corr. of OLR in JJAS"
plot3 = gsn_csm_contour_map_ce(wks, corr_var, rescor)
plot4 = gsn_csm_contour(wks, prob, res_sig)
dum1 = gsn_add_polymarker(wks, plot4, x(k), y(k), pmres)
dum3 = gsn_add_polyline(wks, plot4, x1, y1, gres)
; dum33 = gsn_add_polyline(wks, plot4, x3, y3, gres)
; dum52 = gsn_add_polyline(wks, plot4, x2, y2, gres)
overlay(plot3, plot4)

rescor@gsnLeftString = "(c) Index Corr. of OLR in JJAS"
plot5 = gsn_csm_contour_map(wks, corr_pre, rescor)
plot6 = gsn_csm_contour(wks, prob_pre, res_sig)
dum4 = gsn_add_polyline(wks, plot6, x1, y1, gres)
; dum44 = gsn_add_polyline(wks, plot6, x3, y3, gres)
; dum53 = gsn_add_polyline(wks, plot6, x2, y2, gres)
overlay(plot5, plot6)

pres = True
pres@gsnPanelXWhiteSpacePercent = 5
pres@gsnPanelYWhiteSpacePercent = 5
; pres@gsnPanelLabelBar = True
; pres@lbOrientation = "Horizontal"
; pres@lbLabelFontHeightF = 0.015
; pres@gsnPanelMainString = "Clim. and Std. of monthly Prec."
; pres@gsnStringFont = 26
gsn_panel(wks, (/plot, plot3, plot5/), (/3, 1/), pres)



end