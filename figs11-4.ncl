begin

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

f = addfile("./echam5_comp_diff_pv200.nc", "r")
pv_200_diff = f->pv200
confi_pv = f->pv200_sig

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
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnFillPalette = "MPL_BrBG"
res@cnLevels := (/-0.2,-0.15,-0.1,-0.05,-0.02,0,0.02,0.05,0.1,0.15,0.2/)
res@cnFillColors := (/95,90,85,80,75,-1,-1,45,40,35,30,25/)

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

;;;>-----------------------------------------------------------------------------------------<

wks = gsn_open_wks("pdf", "./figs11-4-final")

;;;>-----------------------------------------------------------------------------------------<

gres = True
gres@gsLineColor = "purple"
gres@gsLineThicknessF = 2

lonw2 = 100
lone2 = 220
lats2 = 45
latn2 = 65
x2 = (/lonw2,lone2,lone2,lonw2,lonw2/)
y2 = (/lats2,lats2,latn2,latn2,lats2/)


res@gsnLeftString = "(d) Diff_dPVdy250 Neg minus Pos"
plot_u200_diff = gsn_csm_contour_map(wks, pv_200_diff, res)
plot_u200_diff_sig = gsn_csm_contour(wks, confi_pv, res_sig)
overlay(plot_u200_diff, plot_u200_diff_sig)

dum2 = gsn_add_polyline(wks, plot_u200_diff_sig, x2, y2, gres)

draw(plot_u200_diff)
frame(wks)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

end
