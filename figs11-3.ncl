begin

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

f = addfile("./echam5_comp_diff_u200.nc", "r") 
u_200_diff = f->u200
confi_u = f->u200_sig
u_200_clim = f->u200_clm

;;;>-----------------------------------------------------------------------------------------<
;;;                                      绘图
;;;>-----------------------------------------------------------------------------------------<

res = True
res@gsnFrame = False
res@gsnDraw = False
res@gsnLeftString = ""
res@gsnRightString = ""
res@gsnCenterString = ""
res@gsnAddCyclic = True
res@tmBorderThicknessF = 2
res@gsnMaximize = True

res_sig = res
res_clm = res

res@vpWidthF = 0.8
res@vpHeightF = 0.4
res@mpShapeMode = "FreeAspect"
res@mpMinLatF = 10
res@mpMaxLatF = 90
res@mpMinLonF = 0
res@mpMaxLonF = 360
res@mpCenterLonF = 180
res@mpFillOn = False
res@mpGeophysicalLineColor = "gray40"
res@mpGeophysicalLineThicknessF = 1.0
res@gsnLeftStringOrthogonalPosF = 0.01
res@gsnLeftStringFontHeightF = 33
res@gsnRightStringOrthogonalPosF = 0.01
res@gsnRightStringFontHeightF = 33

res@tmXBMinorLengthF = 0
res@tmYLMinorLengthF = 0
res@tmXBMajorLengthF = 0.005
res@tmYLMajorLengthF = 0.005
res@tmXBLabelDeltaF = -0.5
res@tmYLLabelDeltaF = -0.5
res@tmXBLabelFontThicknessF = 1.5
res@tmYLLabelFontThicknessF = 1.5
res@tmXBLabelFontHeightF = 0.027
res@tmYLLabelFontHeightF = 0.027
res@tmXBTickSpacingF = 45
res@tmYLTickSpacingF = 20
res@tmXBMajorThicknessF = 1.5
res@tmYLMajorThicknessF = 1.5
res@tmXTOn = False
res@tmYROn = False

res@cnFillOn = True
res@cnLineLabelsOn = False
res@cnLinesOn = False
res@cnFillPalette = "BlWhRe"
res@cnFillOpacityF = 0.8
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnLevels = (/-2,-1.5,-1,-0.5,-0.2,0.2,0,0.5,1,1.5,2/)
res@cnFillColors = (/33,35,37,39,41,-1,-1,56,59,62,65,68/)

res@lbLabelBarOn = True
res@lbOrientation = "Vertical"
res@pmLabelBarWidthF = 0.06
res@lbLabelFontHeightF = 0.025
res@lbBoxEndCapStyle = "TriangleBothEnds"


;;  显著性检验场属性
res_sig@cnFillOn = True
res_sig@lbLabelBarOn = False
res_sig@cnLinesOn = False
res_sig@cnInfoLabelOn = False
res_sig@cnLineLabelsOn = False
res_sig@cnMonoFillPattern = False

res_sig@cnLevelSelectionMode = "ExplicitLevels"
res_sig@cnLevels = (/0.5/)
res_sig@cnFillPatterns = (/-1, 17/)
res_sig@cnFillColors = (/0, 1/)
res_sig@cnFillDotSizeF = 0.004
res_sig@cnFillScaleF = 2.5

res_clm@cnLinesOn = True
res_clm@cnLineLabelsOn = True
res_clm@cnInfoLabelOn = False
res_clm@cnFillOn = False
res_clm@cnLineLabelFontHeightF = 0.008
res_clm@cnLineLabelBackgroundColor = -1
res_clm@cnLineLabelAngleF = 0.0
res_clm@cnLineLabelDensityF = 1
res_clm@cnLineLabelInterval = 1
res_clm@cnLineLabelPerimOn = False
res_clm@cnLineLabelFormat = "0@;*.2f"
res_clm@cnLineDrawOrder = "PostDraw"
res_clm@cnLabelDrawOrder = "PostDraw"
res_clm@cnLevelSelectionMode = "ExplicitLevels"
res_clm@cnLevels = (/20,25,30/)
res_clm@cnLineThicknessF = 2

;;;>-----------------------------------------------------------------------------------------<

wks = gsn_open_wks("pdf", "./figs11-3-final")

;;;>-----------------------------------------------------------------------------------------<

gres = True
gres@gsLineColor = "green"
gres@gsLineThicknessF = 2

lonw1 = 80
lone1 = 170
lats1 = 35
latn1 = 42
x1 = (/lonw1,lone1,lone1,lonw1,lonw1/)
y1 = (/lats1,lats1,latn1,latn1,lats1/)


res@gsnLeftString = "(c) Diff_U250 Neg minus Pos"
plot_u200_diff = gsn_csm_contour_map(wks, u_200_diff, res)
plot_u200_diff_sig = gsn_csm_contour(wks, confi_u, res_sig)
overlay(plot_u200_diff, plot_u200_diff_sig)

plot_u200_clim_model = gsn_csm_contour(wks, u_200_clim, res_clm)
overlay(plot_u200_diff, plot_u200_clim_model)

dum2 = gsn_add_polyline(wks, plot_u200_clim_model, x1, y1, gres)

draw(plot_u200_diff)
frame(wks)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

end
