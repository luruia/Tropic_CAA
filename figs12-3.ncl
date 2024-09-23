begin

h_IN = addfile("./frc_ALL.nc", "r")
IN0 = h_IN->t(:,:,{:50},:)
IN = IN0*24*60*60
copy_VarMeta(IN0, IN)
delete([/h_IN,IN0/])

; Dimensions and sizes:   [lat | 64] x [lon | 128]
; Coordinates:
;             lat: [87.864..-87.864]
;             lon: [ 0..357.1875]

; lon = IN&lon
; lat = IN&lat  

;---X is a two dimensional array dimensioned lat x londims = dimsizes(X)  
X = IN(0,0,:,:)
dims = dimsizes(X) 
x1d = ndtooned(X)      ; convert 2D array to 1D for use in minind
inds = ind_resolve(minind (x1d), dims)    ; convert 1D array back to 2D  
ilat = inds(0,0)        ; select the latitude index where the X array is at its' minimum   
ilon = inds(0,1)        ; select the longitude index where the X array is at its' minimum  
lat_min = X&lat(ilat)   ; insert the latitude index into the lat coordinate variable
lon_min = X&lon(ilon)   ; insert the longitude index into the lon coordinate variable  
print("minimum value located at "+lat_min+", "+lon_min)

tomp = IN(0,:,{toint(lat_min)},{toint(lon_min)})

Q_Model_neg = tomp
Q_Model_neg!0 = "level"
Q_Model_neg&level = (/1000, 950, 900, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5/)

;;;<----------------------------------------------------------------------------->
;;;<                          draw 阔线                                           >
;;;<----------------------------------------------------------------------------->
wks  = gsn_open_wks("pdf","figs12-3")     

res = True
res@vpWidthF = 0.35
res@vpHeightF = 0.5
res@gsnFrame = False
res@gsnDraw = False
res@gsnLeftString = ""
res@gsnRightString = ""
res@gsnCenterString = ""
res@tmBorderThicknessF = 1.5
res@gsnStringFont = 22

res@tmXBMinorLengthF = 0
res@tmYLMinorLengthF = 0
res@tmXBMajorLengthF = 0.005
res@tmYLMajorLengthF = 0.005
res@tmXBLabelDeltaF = -0.5
res@tmYLLabelDeltaF = -0.5
res@tmXBLabelFontThicknessF = 3
res@tmYLLabelFontThicknessF = 3
res@tmXBLabelFontHeightF = 0.025
res@tmYLLabelFontHeightF = 0.025
res@tmXBTickSpacingF = 45
res@tmYLTickSpacingF = 15
res@tmXBMajorThicknessF = 1.5
res@tmYLMajorThicknessF = 1.5
res@tmXTOn = False
res@tmYROn = False

res@trYReverse = True
res@trXMinF = -0.8   
res@trXMaxF = 0.1
res@tmXBMajorThicknessF = 2
res@tmXBMinorLengthF = 0
res@trYMinF = 100   
res@trYMaxF = 1000 
res@tmYLMajorThicknessF = 2
res@tmYLMinorLengthF = 0
res@tiYAxisString = ""
res@gsnLeftString = " "
res@tmXBMode="Explicit"
res@tmXBLabelStride=0
res@tmXBValues=(/-0.8, -0.6, -0.4, -0.2, 0/)
res@tmXBLabels=(/"-0.8", "-0.6", "-0.4", "-0.2", "0"/)

res = res
res@xyLineThicknessF = 2.5
res@xyDashPattern = 16
res@xyLineColor = "blue"
res@lgPerimOn = False
plot_Model_neg  = gsn_csm_xy (wks,Q_Model_neg,Q_Model_neg&level,res) 

draw(plot_Model_neg)
frame(wks)

end
