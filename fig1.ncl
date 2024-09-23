;;;>-----------------------------------------------------------------------------------------<

function to_season(var)     ;;;   返回季节平均的距平
    local dims, n_year, temp, var_season
    begin
        
        dims = dimsizes(var)
        n_year = dims(0)/12

        temp = reshape(var, (/n_year, 12, dims(1), dims(2)/))
        var_season = dim_avg_n(temp(:,5:8,:,:), 1)
        copy_VarCoords(var(0,:,:), var_season(0,:,:))
        var_season!0 = "time"
        var_season&time = ispan(1, n_year, 1)
        var_season = dtrend_msg_n(var_season&time, var_season, True, False, 0)

        return var_season
    end

;;;>-----------------------------------------------------------------------------------------<

function confi(var)     ;;;   返回t检验结果
    local p, dims, confi_var
    begin

        dims = dimsizes(var)
        p = student_t(var@tval, var@nptxy-2)
        confi_var = onedtond( tofloat(p), (/dims(0),dims(1)/) )
        copy_VarCoords(var, confi_var)

        return confi_var
    end

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

begin

;;;>-----------------------------------------------------------------------------------------<

yyyymm_s = 197901
yyyymm_e = 202112

;;;>-----------------------------------------------------------------------------------------<

f = addfile("./olr.mon.mean.nc", "r")
time = cd_calendar(f->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
pre = f->olr(ind_s:ind_e,:,:)
delete([/time, ind_s, ind_e, f/])

pre_jjas = to_season(pre)

;;;>-----------------------------------------------------------------------------------------<

f2 = addfile("./ERA5_2mTSLP_1x1_1940_2023_mon.nc", "r")
time = cd_calendar(f2->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
t2m = short2flt(f2->t2m(ind_s:ind_e,:,:))
dims_t2m = dimsizes(t2m)
delete([/time, ind_s, ind_e, f2/])

t2m_jjas = to_season(t2m)

;;;>-----------------------------------------------------------------------------------------<

f_pc_olr = addfile("./olr_index.nc", "r")
pc_olr = f_pc_olr->pc(yyyymm_s/100-1979:yyyymm_e/100-1979)
pc_olr = pc_olr * (-1.0)

f_pc_sat = addfile("./t2m_index.nc", "r")
pc_sat = f_pc_sat->pc(yyyymm_s/100-1979:yyyymm_e/100-1979)

;;;>-----------------------------------------------------------------------------------------<

corr_pre = escorc(pc_olr, pre_jjas({lat|:},{lon|:},{time|:}))
copy_VarCoords(pre_jjas(0,:,:), corr_pre)
prob_pre = rtest(corr_pre, dimsizes(pc_olr), 0)
copy_VarCoords(pre_jjas(0,:,:), prob_pre)

sat_r = regCoef_n(pc_olr, t2m_jjas, 0, 0)    
copy_VarCoords(t2m_jjas(0,:,:), sat_r)
confi_sat_r = confi(sat_r)

;;;>-----------------------------------------------------------------------------------------<

sig = asciiread("./corr_sig.csv", (/102, 10/), "float")
degree = sig(:,0)
sig_90 = sig(:,3)

N = dimsizes(pc_olr)

nave = 11
run_cor_11 = new(N, "float")
run_cor_11 = -999
run_cor_11@_FillValue = -999
edof_11 = new(N, "float")
edof_11 = -999
edof_11@_FillValue = -999

do m = nave/2, (N-nave/2-1), 1

    run_cor_11(m) = escorc(pc_olr(m-nave/2:m+nave/2), pc_sat(m-nave/2:m+nave/2))

    rpc_olr = esacr(pc_olr(m-nave/2:m+nave/2), 1)
    rpc_sat = esacr(pc_sat(m-nave/2:m+nave/2), 1)
    edof_temp = dimsizes(pc_olr(m-nave/2:m+nave/2))*(1 - rpc_olr(1)*rpc_sat(1))/(1 + rpc_olr(1)*rpc_sat(1))
    min_ind = local_min_1d(abs(edof_temp-degree), False, 0, 1)
    edof_11(m) = sig_90(min_ind)

end do


;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

res = True
res@gsnFrame = False
res@gsnDraw = False
res@gsnLeftString = ""
res@gsnRightString = ""
res@gsnCenterString = ""
; res@tmBorderThicknessF = 3

res_sig = res
res_polar = res

res@mpShapeMode = "FreeAspect"
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

res@cnFillOn = True
res@cnLinesOn = False
res@cnLineLabelsOn = False
res@cnInfoLabelOn = False
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnLevels = (/-0.8,-0.5,-0.3,0,0.3,0.5,0.8/)
res@cnLineDrawOrder = "PostDraw"
res@cnLineThicknessF = 2
res@cnFillOpacityF = 0.7
res@cnFillPalette = "testcmap"
res@cnFillColors = (/72,84,94,-1,-1,115,135,155/)

res@lbLabelBarOn = True
res@lbOrientation = "Vertical"
res@pmLabelBarWidthF = 0.04
res@lbLabelFontHeightF = 0.015
res@lbBoxEndCapStyle = "TriangleBothEnds"

;;;>-----------------------------------------------------------------------------------------<

res_polar@gsnPolar = "NH"
res_polar@mpShapeMode = "FreeAspect"
res_polar@mpMinLatF = 65
res_polar@mpCenterLonF = 180
res_polar@mpFillOn = False
res_polar@mpGeophysicalLineColor = "gray40"
res_polar@mpGeophysicalLineThicknessF = 2.0
res_polar@gsnPolarLabelFontHeightF = 0.015
res_polar@gsnLeftStringFontHeightF = 20
res_polar@gsnLeftStringOrthogonalPosF = 0.093
res_polar@gsnLeftStringParallelPosF = -0.3

res_polar@cnFillOn = True
res_polar@cnLinesOn = False
res_polar@cnLineLabelsOn = False
res_polar@cnInfoLabelOn = False
res_polar@cnFillPalette = "BlWhRe"
res_polar@cnLevelSelectionMode = "ExplicitLevels"
res_polar@cnLevels = (/-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0/)
res_polar@cnFillColors = (/35, 37, 39, 43, 47, -1, -1, 56, 62, 68, 74,80/)

res_polar@lbLabelBarOn = True
res_polar@lbOrientation = "Vertical"
res_polar@pmLabelBarWidthF = 0.04
res_polar@lbLabelFontHeightF = 0.015
res_polar@lbBoxEndCapStyle = "TriangleBothEnds"
res_polar@pmLabelBarOrthogonalPosF = 0.22
res_polar@gsnPolarLabelDistance = 1.08

;;;>-----------------------------------------------------------------------------------------<

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

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

rxy = True
rxy@gsnFrame = False
rxy@gsnDraw = False
rxy@gsnLeftString = "(c) Indices IPDC & AA_SAT"
rxy@gsnLeftStringFontHeightF = 20

rxy@gsnYRefLine = 0 

rxy@trYMaxF = 3
rxy@trYMinF = -3
rxy@trXMaxF = 2022
rxy@trXMinF = 1978

rxy@tmXTMinorOn = False
rxy@tmYRMinorOn = False
rxy@tmXBMinorOn = False
rxy@tmYLMinorOn = False
rxy@tmXBMajorLengthF= -0.005
rxy@tmYLMajorLengthF= -0.005
rxy@tmYLLabelDeltaF = -0.7
rxy@tmXBLabelDeltaF = -0.7
rxy@tmXBLabelFontHeightF = 0.015
rxy@tmYLLabelFontHeightF = 0.015

rxy@tmXMajorGrid = True
rxy@tmXMajorGridThicknessF = 0.15
rxy@tmXMajorGridLineDashPattern = 2
rxy@tmXMajorGridLineColor = "grey"
rxy@tmYMajorGrid = True
rxy@tmYMajorGridThicknessF = 0.15
rxy@tmYMajorGridLineDashPattern = 2
rxy@tmYMajorGridLineColor = "grey"
rxy@tmGridDrawOrder = "PreDraw"

rxy@xyDashPatterns = (/0, 0/)
rxy@xyLineColors = (/"red", "blue"/)
rxy@xyLineThicknesses = (/2, 2/)

rxy@pmLegendDisplayMode = "Always"
rxy@pmLegendWidthF = 0.1
rxy@pmLegendHeightF = 0.043
rxy@pmLegendOrthogonalPosF = -1.19
rxy@pmLegendParallelPosF = 0.15

rxy@lgPerimFill = "SolidFill"
rxy@lgPerimFillColor = "white"
rxy@lgLabelFontHeightF = 0.01
rxy@lgBoxMinorExtentF = 0.25

rxy@lgItemOrder = (/1, 0/)
rxy@xyExplicitLabels = (/"  IPDC","  AA_SAT"/)

rxy@tmXBMode = "Explicit"
rxy@tmXBValues = ispan(1979, 2021, 5)
rxy@tmXBLabels = ispan(1979, 2021, 5)

;;;>-----------------------------------------------------------------------------------------<

rxy2 = True
rxy2@gsnFrame = False
rxy2@gsnDraw = False
rxy2@gsnLeftString = "(d) Run Cor_IPDC & AA_SAT"
rxy2@gsnLeftStringFontHeightF = 20

rxy2@trYMaxF = 1.0
rxy2@trYMinF = -0.5
rxy2@trXMaxF = 2022
rxy2@trXMinF = 1978

rxy2@tmXTMinorOn = False
rxy2@tmYRMinorOn = False
rxy2@tmXBMinorOn = False
rxy2@tmYLMinorOn = False
rxy2@tmXBMajorLengthF= -0.005
rxy2@tmYLMajorLengthF= -0.005
rxy2@tmYLLabelDeltaF = -0.7
rxy2@tmXBLabelDeltaF = -0.7
rxy2@tmXBLabelFontHeightF = 0.015
rxy2@tmYLLabelFontHeightF = 0.015

rxy2@tmXMajorGrid = True
rxy2@tmXMajorGridThicknessF = 0.15
rxy2@tmXMajorGridLineDashPattern = 2
rxy2@tmXMajorGridLineColor = "grey"
rxy2@tmYMajorGrid = True
rxy2@tmYMajorGridThicknessF = 0.15
rxy2@tmYMajorGridLineDashPattern = 2
rxy2@tmYMajorGridLineColor = "grey"
rxy2@tmGridDrawOrder = "PreDraw"

rxy2@tmXBMode = "Explicit"
rxy2@tmXBValues = ispan(1979, 2021, 5)
rxy2@tmXBLabels = ispan(1979, 2021, 5)

rxy2@xyLineThicknessF = 2
rxy2@xyLineColor = "purple"
rxy2@xyCurveDrawOrder = "PostDraw"

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

gres = True
gres@gsLineColor = "blue"
gres@gsLineThicknessF = 3
gres@gsLineDashPattern = 2


lonw = 80
lonc = 180
lone = 260
lats = -15
latn = 15
x1 = (/lonw,lonc,lone,lone,lonc,lonw,lonw/)
y1 = (/lats,lats,lats,latn,latn,latn,lats/)


lonw1 = 230
lone1 = 280
lats1 = 72
latn1 = 80
x2 = (/lonw1,lone1,lone1,lonw1,lonw1/)
y2 = (/lats1,lats1,latn1,latn1,lats1/)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

wks = gsn_open_wks("pdf", "fig1")

;;;>-----------------------------------------------------------------------------------------<

res@vpXF = 0.05
res@vpYF = 0.8
res@vpWidthF = 0.43
res@vpHeightF = 0.25

res@gsnLeftString = "(a) Cor_IPDC & OLR"
plot_olr = gsn_csm_contour_map(wks, corr_pre, res)
plot_olr_sig = gsn_csm_contour(wks, prob_pre, res_sig)
dum1 = gsn_add_polyline(wks, plot_olr_sig, x1, y1, gres)
overlay(plot_olr, plot_olr_sig)

draw(plot_olr)

;;;>-----------------------------------------------------------------------------------------<

res_polar@vpXF = 0.63
res_polar@vpYF = 0.79
res_polar@vpWidthF = 0.25
res_polar@vpHeightF = 0.25

res_sig@cnLevels = (/0.1/)
res_sig@cnFillPatterns = (/17, -1/)
res_sig@cnFillColors = (/1, 0/)
res_sig@cnFillDotSizeF := 0.003
; res_sig@cnFillScaleF := 1.2

res_polar@gsnLeftString = "(b) Reg_SAT onto IPDC"
plot_t2m = gsn_csm_contour_map_polar(wks, sat_r, res_polar)
plot_t2m_sig = gsn_csm_contour(wks, confi_sat_r, res_sig)
gres@gsLineDashPattern = 0
dum2 = gsn_add_polyline(wks, plot_t2m_sig, x2, y2, gres)
overlay(plot_t2m, plot_t2m_sig)

draw(plot_t2m)

;;;>-----------------------------------------------------------------------------------------<

rxy@vpXF = 0.05
rxy@vpYF = 0.45
rxy@vpWidthF = 0.43
rxy@vpHeightF = 0.2

plot_pc = gsn_csm_xy(wks, ispan(1979, 2021, 1), (/pc_olr, pc_sat/), rxy)

draw(plot_pc)

;;;>-----------------------------------------------------------------------------------------<

rxy2@vpXF = 0.56
rxy2@vpYF = 0.45
rxy2@vpWidthF = 0.43
rxy2@vpHeightF = 0.2

plot_run_cor = gsn_csm_xy(wks, ispan(1979, 2021, 1), run_cor_11, rxy2)

rxy2@xyDashPattern = 2
rxy2@xyLineColor = "black"
rxy2@xyLineThicknessF := 1.5
plot_edof = gsn_csm_xy(wks, ispan(1979, 2021, 1), edof_11, rxy2)
overlay(plot_run_cor, plot_edof)

rxy2@xyMarkLineMode = "Markers"
rxy2@xyMarker = 16
rxy2@xyMarkerSizeF = 6
rxy2@xyMarkerColor = "purple"
run_cor_11 = where(run_cor_11 .le. edof_11, run_cor_11@_FillValue, run_cor_11)
plot_run_cor_dot = gsn_csm_xy(wks, ispan(1979, 2021, 1), run_cor_11, rxy2)
overlay(plot_run_cor, plot_run_cor_dot)


pgres = True
pgres@gsFillColor = "grey"
pgres@gsFillOpacityF = 0.3

pgx1 = (/1978,1991,1991,1978,1978/)
pgy1 = (/-1,-1,1,1,-1/)
pg1 = gsn_add_polygon(wks, plot_run_cor, pgx1, pgy1, pgres)
pgx2 = (/2004,2022,2022,2004,2004/)
pgy2 = (/-1,-1,1,1,-1/)
pg2 = gsn_add_polygon(wks, plot_run_cor, pgx2, pgy2, pgres)

draw(plot_run_cor)

frame(wks)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<


end