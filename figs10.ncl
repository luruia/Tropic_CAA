begin

;;;>-----------------------------------------------------------------------------------------<


f_sst = addfile("./clim_diff_sst.nc", "r")
sst = f_sst->sst231
delete(f_sst)

sst_box = sst 
sst_box = 0.0
sst_box({latitude|70:83},{longitude|30:170}) = sst({latitude|70:83},{longitude|30:170}) * (-1.)

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
; res@gsnMaximize = True

res@vpWidthF = 0.8
res@vpHeightF = 0.35
res@mpShapeMode = "FreeAspect"
res@mpMinLatF = 20
res@mpMaxLatF = 90
res@mpMinLonF = 0
res@mpMaxLonF = 190
res@mpCenterLonF = 95
res@mpFillOn = False
res@mpGeophysicalLineColor = "gray40"
res@mpGeophysicalLineThicknessF = 2.0
res@gsnLeftStringOrthogonalPosF = 0.01
res@gsnLeftStringFontHeightF = 30
res@gsnRightStringOrthogonalPosF = 0.01
res@gsnRightStringFontHeightF = 30

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
res@tmXBTickSpacingF = 30
res@tmYLTickSpacingF = 20
res@tmXBMajorThicknessF = 2
res@tmYLMajorThicknessF = 2
res@tmXTOn = False
res@tmYROn = False


res@cnLinesOn = False
res@cnLineLabelsOn = False
res@cnInfoLabelOn = False
res@cnFillOn = True
res@cnFillPalette = "BlWhRe"
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnLevels = (/-1.0,-0.7,-0.5,-0.3,-0.1,0,0.1,0.3,0.5,0.7,1.0/)   ;;; raw
res@cnFillColors = (/33,35,37,39,41,-1,-1,56,59,62,65,68/)
res@cnFillDrawOrder = "PreDraw"
res@lbLabelFontHeightF = 0.02
res@lbLabelBarOn = True
res@lbOrientation = "Vertical"
res@pmLabelBarWidthF = 0.06
res@lbBoxEndCapStyle = "TriangleBothEnds"

lonw13 = 30
lone13 = 170
lats13 = 70
latn13 = 83
x13 = (/lonw13,lone13,lone13,lonw13,lonw13/)
y13 = (/lats13,lats13,latn13,latn13,lats13/)

;;;>-----------------------------------------------------------------------------------------<

wks = gsn_open_wks("pdf", "./figs10")

plot_sst = gsn_csm_contour_map(wks, sst_box, res)

gres = True
gres@gsLineThicknessF = 2
gres@gsLineColor = "red"
dum3 = gsn_add_polyline(wks, plot_sst, x13, y13, gres)

draw(plot_sst)
frame(wks)


;;;>-----------------------------------------------------------------------------------------<

end
