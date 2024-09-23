begin

ff =addfile("./Response_IPDC_ALL.nc", "r")
hgt_Sen = ff->hgt_Sen
u_Sen = ff->u_Sen
v_Sen = ff->v_Sen
hgt_Ctrl = ff->hgt_Ctrl
u_Ctrl = ff->u_Ctrl
v_Ctrl = ff->v_Ctrl

;;
;;<---------------------         PLOT       ------------------------->
wks = gsn_open_wks("pdf", "figs13")  

n = 2
plot_z = new(n,graphic)
plot_uv = new(n,graphic)
dim = new(n,graphic)

res = True
res@gsnFrame = False
res@gsnDraw = False
res@gsnLeftString = ""
res@gsnRightString = ""
res@gsnCenterString = ""
res@gsnAddCyclic = True
res@tmBorderThicknessF = 1.5


res@vpWidthF = 0.8
res@vpHeightF = 0.4
res@gsnLeftStringOrthogonalPosF = 0.01
res@gsnLeftStringFontHeightF = 35
res@gsnRightStringOrthogonalPosF = 0.01
res@gsnRightStringFontHeightF = 35

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
res@tmXBMajorThicknessF = 1.5
res@tmYLMajorThicknessF = 1.5
res@tmXTOn = False
res@tmYROn = False

resh = res
resh@mpShapeMode = "FreeAspect"
resh@mpMinLatF = 10
resh@mpMaxLatF = 90
resh@mpMinLonF = 130
resh@mpMaxLonF = 310
resh@mpCenterLonF = 220
resh@mpFillOn = False
; res@mpLandFillColor = "white"
; res@mpFillDrawOrder = "PostDraw"
; res@mpOutlineDrawOrder = "PreDraw"
resh@mpGeophysicalLineColor = "gray40"
resh@mpGeophysicalLineThicknessF = 0.7

resh@cnLinesOn = False
resh@cnLineLabelsOn = False
resh@cnInfoLabelOn = False
resh@cnFillOn = True
; res@cnFillOpacityF = 0.5
resh@cnLevelSelectionMode = "ExplicitLevels"
resh@lbLabelBarOn = False
resh@cnFillPalette = "MPL_BrBG"
resh@cnLevels = (/ -3.0, -2.0, -1.0, -0.8, -0.6, 0.6, 0.8, 1.0, 2.0, 3.0/)
resh@cnLevels = (/ -3.0, -2.0, -1.0, -0.8, -0.6, 0.2, 0.4, 0.6, 0.8, 1.0/)
resh@cnFillColors = (/100,95,90,85,80,-1,45,40,35,30,25/)


resh@gsnLeftString = "(a) Exp_Ctrl"
plot_z(0) = gsn_csm_contour_map(wks, hgt_Ctrl, resh)  
resh@gsnLeftString = "(b) Exp_Arctic_cooling"
plot_z(1) = gsn_csm_contour_map(wks, hgt_Sen, resh) 

;;  WIND
res_uv              = res
res_uv@vcGlyphStyle = "CurlyVector"
res_uv@vcMinDistanceF = 0.03
res_uv@vcLineArrowThicknessF = 1.5

res_uv@vcRefAnnoOn = True
res_uv@vcRefLengthF = 0.035
res_uv@vcRefAnnoBackgroundColor = "white"
res_uv@vcRefAnnoPerimOn = True   ;关闭边框
res_uv@vcRefAnnoFontHeightF = 0.012
res_uv@vcRefAnnoString1On =   False ;设置标注箭头上方不显示注释
res_uv@vcRefAnnoString2On = True   ;设置标注箭头下方显示注释
res_uv@vcRefAnnoFontThicknessF = 1.5
res_uv@vcRefAnnoOrthogonalPosF = -0.11
res_uv@vcRefAnnoParallelPosF = 0.999
res_uv@vcVectorDrawOrder = "PostDraw"
res_uv@vcLineArrowHeadMaxSizeF = 0.01
res_uv@vcLineArrowHeadMinSizeF = 0.01 

;; 1-plot
refmag_uv = 0.1
res_uv@vcRefMagnitudeF = refmag_uv
res_uv@vcRefAnnoString2 = refmag_uv+" m/s"
min_wind = 0.02 
u_Ctrl = where(sqrt((/u_Ctrl/)^2 + (/v_Ctrl/)^2) .gt. min_wind, u_Ctrl, 0)
v_Ctrl = where(sqrt((/u_Ctrl/)^2 + (/v_Ctrl/)^2) .gt. min_wind, v_Ctrl, 0) 
plot_uv(0) = gsn_csm_vector(wks, u_Ctrl, v_Ctrl, res_uv) 
overlay(plot_z(0), plot_uv(0))

u_Sen = where(sqrt((/u_Sen/)^2 + (/v_Sen/)^2) .gt. min_wind, u_Sen, 0)
v_Sen = where(sqrt((/u_Sen/)^2 + (/v_Sen/)^2) .gt. min_wind, v_Sen, 0) 
plot_uv(1) = gsn_csm_vector(wks, u_Sen, v_Sen, res_uv) 
overlay(plot_z(1), plot_uv(1))


resP                  = True                   ; modify the panel plot
resP@gsnPanelLabelBar = True
resP@lbOrientation = "Horizontal"
resP@cnLabelBarEndStyle = "TriangleBothEnds"
; resP@lbFillOpacityF = 0.8
resP@pmLabelBarHeightF = 0.04
resP@pmLabelBarWidthF = 0.54
resP@pmLabelBarParallelPosF = .008
resP@pmLabelBarOrthogonalPosF = -0.003
resP@lbLabelFontHeightF = 0.01
resP@gsnPanelYWhiteSpacePercent = 2
resP@gsnPanelXWhiteSpacePercent = 2
resP@lbBoxEndCapStyle = "TriangleBothEnds"

gsn_panel(wks, plot_z, (/1, 2/), resP) 


end


