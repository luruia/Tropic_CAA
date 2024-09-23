begin

;;;>-----------------------------------------------------------------------------------------<

yyyymm_s = 197901
yyyymm_e = 202112

f = addfile("./gpcp_v02r03_monthly_197901_202112.nc", "r")
time = cd_calendar(f->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
pre = f->precip(ind_s:ind_e,:,:)

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
temp = pre_std({latitude|-15:25},{longitude|60:160})
lat_temp = temp&latitude
lon_temp = temp&longitude
nmax = local_max(temp, False, 0)
x = lon_temp(nmax@xi)
y = lat_temp(nmax@yi)
print(x)
print(y)
k = 6

print(pre_std({y(k)},{x(k)}))
pc_max = pre_jjas(:, {y(k)},{x(k)})
corr_var = escorc(pc_max, pre_jjas({latitude|:},{longitude|:},{time|:}))
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

pre_wgt = wgt_areaave2(pre_jjas(:,{lats:latn},{lonw:lone}), corr_var({lats:latn},{lonw:lone}), 0)
pre_wgt = pre_wgt * (-1.)
pre_wgt = dim_standardize(pre_wgt, 1)


corr_pre = escorc(pre_wgt, pre_jjas({latitude|:},{longitude|:},{time|:}))
copy_VarCoords(pre_jjas(0,:,:), corr_pre)
; printVarSummary(corr_var)
prob_pre = rtest(corr_pre, n_year, 0)
copy_VarCoords(pre_jjas(0,:,:), prob_pre)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

res = True
res@gsnFrame = False
res@gsnDraw = False
res@gsnLeftString = ""
res@gsnRightString = ""
res@gsnCenterString = ""
res@gsnAddCyclic = True
res@tmBorderThicknessF = 2

resm = res
res_sig = res

res@mpMinLatF = -40
res@mpMaxLatF = 40
res@mpCenterLonF = 180
res@mpFillOn = False
res@mpGeophysicalLineColor = "gray40"
res@mpGeophysicalLineThicknessF = 2.0
res@gsnLeftStringFontHeightF = 20
res@gsnRightStringFontHeightF = 20

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
res@tmXBTickSpacingF = 45
res@tmYLTickSpacingF = 20
res@tmXBMajorThicknessF = 2
res@tmYLMajorThicknessF = 2
res@tmXTOn = False
res@tmYROn = False

rescor = res

res@cnLinesOn = False
res@cnLineLabelsOn = False
res@cnInfoLabelOn = False
res@cnFillOn = True
res@cnFillPalette = "MPL_PuOr"
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnLevels := (/1,2,3,4,5,6/)
res@cnFillColors = (/-1,50,40,30,20,10,5/)

res@lbLabelBarOn = True
res@lbOrientation = "Vertical"
res@pmLabelBarWidthF = 0.03
res@lbLabelFontHeightF = 0.015
res@lbBoxEndCapStyle = "TriangleBothEnds"

resm@cnLinesOn = True
resm@cnLineLabelsOn = True
resm@cnInfoLabelOn = False
resm@cnFillOn = False
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
resm@cnLevels = (/5, 10/)
resm@gsnContourZeroLineThicknessF = 0.0

rescor@cnFillOn = True
rescor@cnLinesOn = False
rescor@cnLineLabelsOn = False
rescor@cnInfoLabelOn = False
rescor@cnLevelSelectionMode = "ExplicitLevels"
rescor@cnLevels = (/-0.8,-0.5,-0.3,0,0.3,0.5,0.8/)
rescor@cnLineDrawOrder = "PostDraw"
rescor@cnLineThicknessF = 2
rescor@cnFillOpacityF = 0.7
rescor@lbLabelBarOn = True
rescor@lbOrientation = "Vertical"
rescor@pmLabelBarWidthF = 0.03
rescor@lbLabelFontHeightF = 0.015
rescor@lbBoxEndCapStyle = "TriangleBothEnds"
rescor@cnFillPalette = "testcmap"
rescor@cnFillColors = (/155,135,115,-1,-1,94,84,72/)

res_sig@cnFillOn = True
res_sig@lbLabelBarOn = False
res_sig@cnLinesOn = False
res_sig@cnInfoLabelOn = False
res_sig@cnLineLabelsOn = False
res_sig@cnMonoFillPattern = False
res_sig@cnLevelSelectionMode = "ExplicitLevels"
res_sig@cnLevels = (/0.05/)
res_sig@cnFillPatterns = (/8, -1/)
res_sig@cnFillColors = (/1, 0/)
res_sig@cnFillDotSizeF = 0
res_sig@cnFillDrawOrder = "PostDraw"

pmres = True
pmres@gsMarkerIndex = 12
pmres@gsMarkerSizeF = 0.02
pmres@gsMarkerThicknessF = 2.5
pmres@gsMarkerColor = "blue"

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

wks = gsn_open_wks("pdf", "figs3")

res@gsnLeftString = "(a) Std of Pr in JJAS"
plot = gsn_csm_contour_map(wks, pre_std, res)
plot2 = gsn_csm_contour(wks, pre_clm, resm)
dum2 = gsn_add_polyline(wks, plot, x1, y1, gres)
dum01 = gsn_add_polymarker(wks, plot2, x(k), y(k), pmres)
overlay(plot, plot2)

;;;>-----------------------------------------------------------------------------------------<

rescor@gsnLeftString = "(b) Point Cor_Pr in JJAS"
plot3 = gsn_csm_contour_map_ce(wks, corr_var, rescor)
plot4 = gsn_csm_contour(wks, prob, res_sig)
dum1 = gsn_add_polymarker(wks, plot4, x(k), y(k), pmres)
dum3 = gsn_add_polyline(wks, plot4, x1, y1, gres)
overlay(plot3, plot4)

;;;>-----------------------------------------------------------------------------------------<

rescor@gsnLeftString = "(c) Index Cor_Pr in JJAS"
plot5 = gsn_csm_contour_map(wks, corr_pre, rescor)
plot6 = gsn_csm_contour(wks, prob_pre, res_sig)
dum4 = gsn_add_polyline(wks, plot6, x1, y1, gres)
overlay(plot5, plot6)

;;;>-----------------------------------------------------------------------------------------<

pres = True
pres@gsnPanelXWhiteSpacePercent = 5
pres@gsnPanelYWhiteSpacePercent = 5
gsn_panel(wks, (/plot, plot3, plot5/), (/3, 1/), pres)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

end