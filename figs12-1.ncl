begin

h_IN = addfile("./frc_ALL.nc", "r")

IN0 = h_IN->t(0,1,:,:)
IN = IN0*30*24*60*60
copy_VarCoords(IN0, IN)
printMinMax(IN,True)

wks = gsn_open_wks("pdf", "figs12-1")
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
res@tmBorderThicknessF = 1.5
res@gsnStringFont = 22

res_sig = res
res_tn = res
res850 = res
res_pre = res
res_uv = res
res2 = res

res@vpWidthF = 0.8
res@vpHeightF = 0.4
res@mpShapeMode = "FreeAspect"
res@mpMinLatF = -15
res@mpMaxLatF = 90
; res@mpMinLonF = 130
; res@mpMaxLonF = 310
res@mpCenterLonF = 180
res@mpFillOn = False
; res@mpLandFillColor = "white"
; res@mpFillDrawOrder = "PostDraw"
; res@mpOutlineDrawOrder = "PreDraw"
res@mpGeophysicalLineColor = "gray40"
res@mpGeophysicalLineThicknessF = 1.0
res@gsnLeftStringOrthogonalPosF = 0.01
res@gsnLeftStringFontHeightF = 25
res@gsnRightStringOrthogonalPosF = 0.01

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
res@tmYLTickSpacingF = 15
res@tmXBMajorThicknessF = 1.5
res@tmYLMajorThicknessF = 1.5
res@tmXTOn = False
res@tmYROn = False

resh = res

resh@cnLinesOn = False
resh@cnLineLabelsOn = False
resh@cnInfoLabelOn = False
resh@cnFillOn = True
; res@cnFillOpacityF = 0.5
resh@cnLevelSelectionMode = "ExplicitLevels"
resh@cnFillPalette = "BlWhRe"
resh@cnLevels = (/-5.0, -3.0, -2.0, -1.0,-0.5, 0.5, 1.0, 2.0, 3.0, 5.0/)
resh@cnFillColors = (/30,33,36,39,42,-1,58,61,66,70,80/)

resh@lbLabelBarOn = True
resh@lbOrientation = "Vertical"
resh@pmLabelBarWidthF = 0.05
resh@lbLabelFontHeightF = 0.017
resh@lbBoxEndCapStyle = "TriangleBothEnds"

resh@gsnLeftString = "(b) Exp_Arctic_cooling"
plot_heat_IN = gsn_csm_contour_map(wks, IN, resh)  

draw(plot_heat_IN)
frame(wks)


end
