begin

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

f = addfile("./echam5_comp_reg_neg.nc", "r") 
; f = addfile("./echam5_comp_reg_pos.nc", "r") 
hgt_200_r = f->hgt
confi_z = f->hgt_sig
u_200_r = f->u200
v_200_r = f->v200

;;;>-----------------------------------------------------------------------------------------<
;;;                绘图
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
res_uv = res

res@vpWidthF = 0.8
res@vpHeightF = 0.4
res@mpShapeMode = "FreeAspect"
res@mpMinLatF = 10
res@mpMaxLatF = 90
res@mpMinLonF = 130
res@mpMaxLonF = 310
res@mpCenterLonF = 220
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
res@tmXBTickSpacingF = 30
res@tmYLTickSpacingF = 20
res@tmXBMajorThicknessF = 1.5
res@tmYLMajorThicknessF = 1.5
res@tmXTOn = False
res@tmYROn = False

res@cnLinesOn = False
res@cnLineLabelsOn = False
res@cnInfoLabelOn = False
res@cnFillOn = True
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnFillPalette = "MPL_BrBG"
res@cnFillColors = (/100,95,90,85,80,-1,-1,45,40,35,30,25/)

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


res_uv@vcGlyphStyle = "CurlyVector"
res_uv@vcMinDistanceF = 0.03
res_uv@vcLineArrowThicknessF = 0.8
; res_uv@vcLineArrowColor = "grey"

res_uv@vcRefAnnoOn = True
res_uv@vcRefLengthF = 0.035
res_uv@vcRefAnnoBackgroundColor = "white"
res_uv@vcRefAnnoPerimOn = True   ;关闭边框
res_uv@vcRefAnnoFontHeightF = 0.012
res_uv@vcRefAnnoString1On =   False ;设置标注箭头上方不显示注释
res_uv@vcRefAnnoString2On = True   ;设置标注箭头下方显示注释
res_uv@vcRefAnnoFontThicknessF = 1.5
res_uv@vcRefAnnoOrthogonalPosF = -0.15
res_uv@vcRefAnnoParallelPosF = 0.999
res_uv@vcVectorDrawOrder = "PostDraw"
res_uv@vcLineArrowHeadMaxSizeF = 0.015
res_uv@vcLineArrowHeadMinSizeF = 0.015

;;;>-----------------------------------------------------------------------------------------<

wks = gsn_open_wks("pdf", "./figs11-1-final")
    
;;;>-----------------------------------------------------------------------------------------<

res@gsnLeftString = "(a) Neg_comp Reg_UVZ250"
res@cnLevels := (/-15,-10,-6,-3,-1,0,1,3,6,10,15/)
plot_h200 = gsn_csm_contour_map(wks, hgt_200_r, res)
plot_h200_sig = gsn_csm_contour(wks, confi_z, res_sig)
overlay(plot_h200, plot_h200_sig)

res_uv@vcRefMagnitudeF := 1.0 ;;; raw
res_uv@vcRefAnnoString2 := "1.0 m/s"
min200 = 0.2     ;;  mask小值  raw
u_200_r = where(sqrt((/u_200_r/)^2 + (/v_200_r/)^2) .gt. min200, u_200_r, 0)
v_200_r = where(sqrt((/u_200_r/)^2 + (/v_200_r/)^2) .gt. min200, v_200_r, 0)
plot_uv200 = gsn_csm_vector(wks, u_200_r, v_200_r, res_uv)
overlay(plot_h200, plot_uv200)

draw(plot_h200)
frame(wks)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

end
