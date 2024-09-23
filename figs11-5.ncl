begin

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

data = asciiread("./echam5_index.csv", (/16, 2/), "float")
index_jet = data(:,0)
index_pv  = data(:,1)

exp_N = dimsizes(index_jet)

;;;>-----------------------------------------------------------------------------------------<

neg_end = exp_N/2-1
pos_sta = exp_N/2

jet_asia_mean1 = avg(index_jet(:neg_end))
jet_asia_std1 = stddev(index_jet(:neg_end))
jet_asia_mean2 = avg(index_jet(pos_sta:exp_N-1))
jet_asia_std2 = stddev(index_jet(pos_sta:exp_N-1))

pv_np_mean1 = avg(index_pv(:neg_end))
pv_np_std1 = stddev(index_pv(:neg_end))
pv_np_mean2 = avg(index_pv(pos_sta:exp_N-1))
pv_np_std2 = stddev(index_pv(pos_sta:exp_N-1))

;;;>-----------------------------------------------------------------------------------------<

res = True
res@gsnFrame = False
res@gsnDraw = False
res@gsnLeftString = ""
res@gsnRightString = ""
res@gsnCenterString = ""
res@tmBorderThicknessF = 2
res@tiYAxisFontHeightF = 0.03
res@vpWidthF = 0.45
res@vpHeightF = 0.35

res@tmXBMinorLengthF = 0
res@tmYLMinorLengthF = 0
res@tmXBMajorLengthF = 0.
res@tmYLMajorLengthF = 0.
res@tmXBLabelDeltaF = -0.5
res@tmYLLabelDeltaF = -0.5
res@tmXBLabelFontThicknessF = 1.5
res@tmYLLabelFontThicknessF = 1.5
res@tmXBLabelFontHeightF = 0.03
res@tmYLLabelFontHeightF = 0.03
res@tmXBMajorThicknessF = 0
res@tmYLMajorThicknessF = 0
res@tmXTOn = False
res@tmYROn = False

res@trXMaxF = 2.7
res@trXMinF = 0.3
res@tmXBMode = "Explicit"
res@tmXBValues = (/1,2/)
res@tmXBLabels = (/"Neg-run","Pos-run"/)
res@tmYLMode = "Explicit"


res@gsnXYBarChart = True
res@gsnXYBarChartBarWidth = 0.7
; res@gsnXYBarChartColors = "blue"
res@gsnAboveYRefLineColor = "orange"
res@gsnBelowYRefLineColor = "deepskyblue"
res@xyLineColor = "transparent"

;;;>-----------------------------------------------------------------------------------------<

wks = gsn_open_wks("pdf", "./figs11-5")

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

; res@tmYLPrecision = 3
res@tiYAxisString = "U250"
res@gsnYRefLine = avg(index_jet)
res@trYMaxF = avg(index_jet)+0.5
res@trYMinF = avg(index_jet)-0.5
res@tmYLValues = (/24.8,25.2,25.6/)
res@tmYLLabels = (/"24.7","25.2","25.6"/)
plot1 = gsn_csm_xy(wks, (/1,2/), (/jet_asia_mean1,jet_asia_mean2/), res)

; res@tmYLPrecision = 2
res@tiYAxisString = "dPVdy250"
res@gsnYRefLine := avg(index_pv)
res@trYMaxF = avg(index_pv)+0.025
res@trYMinF = avg(index_pv)-0.025
res@tmYLValues := (/0.79,0.81,0.83/)
res@tmYLLabels := (/"0.79","0.81","0.83"/)
plot2 = gsn_csm_xy(wks, (/1,2/), (/pv_np_mean1,pv_np_mean2/), res)

;;;>-----------------------------------------------------------------------------------------<

polyres = True
polyres@gsMarkerIndex = 1 
polyres@gsMarkerSizeF = .02
polyres@gsLineColor = "red"
polyres@gsLineThicknessF = 1.5

l1dum1 = gsn_add_polyline(wks, plot1, (/1,1/), (/jet_asia_mean1-jet_asia_std1, jet_asia_mean1+jet_asia_std1/), polyres)
l2dum1 = gsn_add_polyline(wks, plot1, (/1-.15,1+.15/), (/jet_asia_mean1+jet_asia_std1, jet_asia_mean1+jet_asia_std1/), polyres)
l3dum1 = gsn_add_polyline(wks, plot1, (/1-.15,1+.15/), (/jet_asia_mean1-jet_asia_std1, jet_asia_mean1-jet_asia_std1/), polyres)
l1dum2 = gsn_add_polyline(wks, plot1, (/2,2/), (/jet_asia_mean2-jet_asia_std2, jet_asia_mean2+jet_asia_std2/), polyres)
l2dum2 = gsn_add_polyline(wks, plot1, (/2-.15,2+.15/), (/jet_asia_mean2+jet_asia_std2, jet_asia_mean2+jet_asia_std2/), polyres)
l3dum2 = gsn_add_polyline(wks, plot1, (/2-.15,2+.15/), (/jet_asia_mean2-jet_asia_std2, jet_asia_mean2-jet_asia_std2/), polyres)

l1dum3 = gsn_add_polyline(wks, plot2, (/1,1/), (/pv_np_mean1-pv_np_std1, pv_np_mean1+pv_np_std1/), polyres)
l2dum3 = gsn_add_polyline(wks, plot2, (/1-.15,1+.15/), (/pv_np_mean1+pv_np_std1, pv_np_mean1+pv_np_std1/), polyres)
l3dum3 = gsn_add_polyline(wks, plot2, (/1-.15,1+.15/), (/pv_np_mean1-pv_np_std1, pv_np_mean1-pv_np_std1/), polyres)
l1dum4 = gsn_add_polyline(wks, plot2, (/2,2/), (/pv_np_mean2-pv_np_std2, pv_np_mean2+pv_np_std2/), polyres)
l2dum4 = gsn_add_polyline(wks, plot2, (/2-.15,2+.15/), (/pv_np_mean2+pv_np_std2, pv_np_mean2+pv_np_std2/), polyres)
l3dum4 = gsn_add_polyline(wks, plot2, (/2-.15,2+.15/), (/pv_np_mean2-pv_np_std2, pv_np_mean2-pv_np_std2/), polyres)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

pres = True
pres@gsnPanelXWhiteSpacePercent = 3
pres@gsnPanelYWhiteSpacePercent = 5
gsn_panel(wks, (/plot1, plot2/), (/1, 2/), pres)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

end