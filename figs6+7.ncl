load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/run_cor.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

;;;>-----------------------------------------------------------------------------------------<

undef("runave_lr")
function runave_lr(var, nave)     ;;;   分离空间场[time; lat; lon]的年际、年代际部分。
    local var_mean, var_year, m, dims
    begin
        dims = dimsizes(var)
        var_mean = runave_n_Wrap(var, nave, 0, 0)

        do m = 0, (nave/2 - 1), 1
            var_mean(m,:,:) = dim_avg_n_Wrap(var(0:nave/2+m,:,:), 0)
            var_mean(dims(0)-1-m,:,:) = dim_avg_n_Wrap(var(dims(0)-nave/2 -1 -m:,:,:), 0)
        end do

        var_year = var - var_mean
        copy_VarMeta(var, var_year)
        copy_VarMeta(var, var_mean)

        return [/var_mean, var_year/]
    end

;;;>-----------------------------------------------------------------------------------------<

undef("to_season")
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
        ; var_season = dtrend_msg_n(var_season&time, var_season, False, False, 0)

        return var_season
    end

;;;>-----------------------------------------------------------------------------------------<

undef("to_clim")
function to_clim(var)     ;;;   返回季节平均的气候态
    local dims, n_year, temp, var_season, var_clim
    begin
        
        dims = dimsizes(var)
        n_year = dims(0)/12

        temp = reshape(var, (/n_year, 12, dims(1), dims(2)/))
        var_season = dim_avg_n(temp(:,5:8,:,:), 1)
        copy_VarCoords(var(0,:,:), var_season(0,:,:))
        var_season!0 = "time"
        var_season&time = ispan(1, n_year, 1)
        var_season = dtrend_msg_n(var_season&time, var_season, False, False, 0)
        var_clim = dim_avg_n_Wrap(var_season, 0)

        return var_clim
    end

;;;>-----------------------------------------------------------------------------------------<

undef("confi")
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

undef("tnwaf")
function tnwaf(hgt_r, u_clm, v_clm)     ;;;   返回t检验结果
    local zvarorg1, cuvar, cvvar, level, gc, ga, re, sclhgt, pi, f, ilat, nlat, lon, lat, nlon, coslat, \
            coslattmp, ftmp, cumag, psidev, dpsidevdlon, ddpsidevdlonlon, dpsidevdlat, ddpsidevdlonlat, ddpsidevdlatlat, \
            xuterm, xvterm, yvterm, fx, fy
    begin

        zvarorg1 = hgt_r   ;;  异常场

        cuvar = u_clm     ;;   气候态
        cvvar = v_clm

        ; lon = hgt_r&lon
        ; lat = hgt_r&lat
        ; lon = hgt_r&g0_lon_3
        ; lat = hgt_r&g0_lat_2
        lon = hgt_r&longitude
        lat = hgt_r&latitude
        nlon = dimsizes(lon)
        nlat = dimsizes(lat)

        level = 250

        ;;;>-----------------------------------------------------------------------------------------<

        gc = 290   ;;  Gas constant
        ga = 9.80665   ;;  Gravitational acceleration
        re = 6378388    ;;  Radius of the earth
        sclhgt = 8000.   ;; scale height
        pi = atan(1.0)*4.   ;; pi

        f =  2.*2.*pi/(60.*60.*24.)*sin(pi/180. * lat(:))   ;; Coriolis parameter
        f!0 = "lat"
        f&lat = lat
        f@_FillValue = doubletofloat(zvarorg1@_FillValue)

        ; missing for 10S - 10N
        do ilat = 0, nlat-1
            if (abs(lat(ilat) ).lt. 30. ) then
                f(ilat)= f@_FillValue
            end if
        end do


        coslat = cos(lat(:)*pi/180.)   ;; cosine

        coslattmp = conform_dims(dimsizes(zvarorg1), coslat, 0)
        ftmp = conform_dims(dimsizes(zvarorg1), f, 0)

        ; magnitude of climatological wind
        cumag = sqrt(cuvar^2 + cvvar^2)
        cumag@_FillValue = cuvar@_FillValue
        cumag = where(cumag .gt. 0, cumag, cumag@_FillValue)
        copy_VarCoords(cuvar,cumag)

        ; QG steam function for anomaly
        psidev = zvarorg1*ga /ftmp
        copy_VarCoords(zvarorg1, psidev)

        ;dpsidev/dlon
        dpsidevdlon = center_finite_diff_n(psidev, lon*pi/180., True, 0, 1)

        ;ddpsidev/dlonlon
        ddpsidevdlonlon = center_finite_diff_n(dpsidevdlon, lon*pi/180., True, 0, 1)

        ;dpsidev/dlat
        dpsidevdlat = center_finite_diff_n(psidev, lat*pi/180., False, 0, 0)

        ;ddpsidev/dlonlat
        ddpsidevdlonlat = center_finite_diff_n(dpsidevdlon, lat*pi/180., False, 0, 0)

        ;ddpsidev/dlatdlat
        ddpsidevdlatlat = center_finite_diff_n(dpsidevdlat, lat*pi/180., False, 0, 0)

        xuterm = (dpsidevdlon*dpsidevdlon - psidev*ddpsidevdlonlon)
        xvterm = (dpsidevdlon*dpsidevdlat - psidev*ddpsidevdlonlat)
        ;yuterm = xvterm
        yvterm = (dpsidevdlat*dpsidevdlat - psidev*ddpsidevdlatlat)


        ;x-component of (38)
        fx = level/1000./(2.*cumag*re*re)*( cuvar/coslattmp * xuterm + cvvar * xvterm)

        ;y-component 
        fy = level/1000./(2.*cumag*re*re)*( cuvar*xvterm + coslattmp*cvvar*yvterm)

        ; for output
        fx!0 = "lat"
        fx&lat = lat
        fx!1 = "lon"
        fx&lon = lon

        fy!0 = "lat"
        fy&lat = lat
        fy!1 = "lon"
        fy&lon = lon

        fx@units = "m^2/s^2"
        fy@units = "m^2/s^2"

        return [/fx, fy/]
    end

;;;>-----------------------------------------------------------------------------------------<

undef("cal_rws2")
function cal_rws2(u_anom, v_anom, u_clim, v_clim)     ;;;   返回RWS
    local dims,nlat,mlon,uvmsg,dv_anom,vr_anom,dv_clim,vr_clim,f,f_2d,abo_vor_clim,uvd_anom, \
            uvdu_anom,uvdv_anom,uvd_clim,uvdu_clim,uvdv_clim,u1,v1,dv1,u2,v2,dv2,RWS
    begin

        dims = dimsizes(u_anom)
        nlat  = dims(0)                                       ; dimensions
        mlon  = dims(1)
        uvmsg = u_anom@_FillValue

        dv_anom = new ( (/nlat,mlon/), "float", uvmsg )
        vr_anom = new ( (/nlat,mlon/), "float", uvmsg )
        uv2vrdvf(u_anom, v_anom, vr_anom, dv_anom)
        copy_VarMeta(u_anom, dv_anom)
        copy_VarMeta(u_anom, vr_anom)

        dv_clim = new ( (/nlat,mlon/), "float", uvmsg )
        vr_clim = new ( (/nlat,mlon/), "float", uvmsg )
        uv2vrdvf(u_clim, v_clim, vr_clim, dv_clim)
        copy_VarMeta(u_clim, dv_clim)
        copy_VarMeta(u_clim, vr_clim)
        ; printVarSummary(vr_clim)

        f = coriolis_param(v_clim&latitude)
        f_2d = conform(u_clim, f, (/0/))
        abo_vor_clim = vr_clim + f_2d   ;;;   异常绝对涡度

        uvd_anom = dv2uvF_Wrap(dv_anom)
        uvdu_anom = uvd_anom(0,:,:)
        uvdv_anom = uvd_anom(1,:,:)
        copy_VarMeta(u_anom, uvdu_anom)
        copy_VarMeta(u_anom, uvdv_anom)

        uvd_clim = dv2uvF_Wrap(dv_clim)
        uvdu_clim = uvd_clim(0,:,:)
        uvdv_clim = uvd_clim(1,:,:)
        copy_VarMeta(u_anom, uvdu_clim)
        copy_VarMeta(u_anom, uvdv_clim)

        u1 = uvdu_anom * abo_vor_clim
        v1 = uvdv_anom * abo_vor_clim
        dv1 = uv2dvF_Wrap(u1, v1)

        u2 = uvdu_clim * vr_anom
        v2 = uvdv_clim * vr_anom
        dv2 = uv2dvF_Wrap(u2, v2)

        RWS = - dv1 - dv2
        RWS = RWS * 10^11
        copy_VarMeta(u_anom, RWS)

        return RWS
    end

;>---------------------------------------------------------------------------------------<

undef("cal_rws")
function cal_rws(u_anom, v_anom)     ;;;   返回RWS
    local dims,lat,lon,nlat,mlon,uvmsg,dv,vr,uvd,uvdu,uvdv,dv_d,vr_d,dv_clm,vr_clm,uvd_clm, \
            uvdu_clm,uvdv_clm,dv_d_clm,vr_d_clm,f,f_2d,f_3d,abo_vor,abo_vor_clm,dlon,dlat, \
            dvordx,dvordy,dvordx_clm,dvordy_clm,nl,ml,dx,dy,S1,S2,S3,S4,RWS,u_clim,v_clim
    begin

        u_clim = dim_avg_n_Wrap(u_anom, 0)
        v_clim = dim_avg_n_Wrap(v_anom, 0)
        u_anom = dim_rmvmean_n_Wrap(u_anom, 0)
        v_anom = dim_rmvmean_n_Wrap(v_anom, 0)

        dims = dimsizes(u_anom)
        uvmsg = u_anom@_FillValue
        ; lat = u_anom&lat
        ; lon = u_anom&lon
        ; lat = u_anom&g0_lat_2
        ; lon = u_anom&g0_lon_3
        lat = u_anom&latitude
        lon = u_anom&longitude
        nlat = dimsizes(lat)
        mlon = dimsizes(lon)

        dv = uv2dv_cfd(u_anom, v_anom, lat, lon, 3)
        vr = uv2vr_cfd(u_anom, v_anom, lat, lon, 3)
        copy_VarMeta(u_anom, dv)
        copy_VarMeta(u_anom, vr)

        uvd = dv2uvF_Wrap(dv)   ;;;   异常辐散风
        uvdu = uvd(0,:,:,:)
        uvdv = uvd(1,:,:,:)
        copy_VarMeta(u_anom, uvdu)
        copy_VarMeta(u_anom, uvdv)

        dv_d = uv2dv_cfd(uvdu, uvdv, lat, lon, 1)
        vr_d = uv2vr_cfd(uvdu, uvdv, lat, lon, 1)
        copy_VarMeta(u_anom, dv_d)
        copy_VarMeta(u_anom, vr_d)


        dv_clm = uv2dv_cfd(u_clim, v_clim, lat, lon, 3)
        vr_clm = uv2vr_cfd(u_clim, v_clim, lat, lon, 3)
        copy_VarMeta(u_clim, dv_clm)
        copy_VarMeta(u_clim, vr_clm)


        uvd_clm = dv2uvF_Wrap(dv_clm)   ;;;   气候态辐散风
        uvdu_clm = uvd_clm(0,:,:)
        uvdv_clm = uvd_clm(1,:,:)
        copy_VarMeta(u_clim, uvdu_clm)
        copy_VarMeta(u_clim, uvdv_clm)

        dv_d_clm = uv2dv_cfd(uvdu_clm, uvdv_clm, lat, lon, 1)
        vr_d_clm = uv2vr_cfd(uvdu_clm, uvdv_clm, lat, lon, 1)
        copy_VarMeta(u_clim, dv_d_clm)
        copy_VarMeta(u_clim, vr_d_clm)

        f = coriolis_param(lat)
        f_2d = conform(u_clim, f, (/0/))
        f_3d = conform(u_anom, f, (/1/))
        abo_vor = vr + f_3d   ;;;   异常绝对涡度
        abo_vor_clm = vr_clm + f_2d   ;;;   气候态绝对涡度

        ;>---------------------------------------------------------------------------------------<
        ;>---------------------------------------------------------------------------------------<

        dlon = (lon(2)-lon(1)) * 0.0174533
        dlat = (lat(2)-lat(1)) * 0.0174533

        dvordx = new ( (/dims(0), nlat, mlon/), "float", uvmsg )
        dvordy = new ( (/dims(0), nlat, mlon/), "float", uvmsg )
        dvordx_clm = new ( (/nlat, mlon/), "float", uvmsg )
        dvordy_clm = new ( (/nlat, mlon/), "float", uvmsg )

        do nl = 0, nlat-1  
            dx = 6378388.*cos(0.0174533*lat(nl)) * dlon 
            dvordx(:,nl:nl,:) = center_finite_diff_n(abo_vor(:,nl:nl,:), dx, True, 0, 2)
            dvordx_clm(nl:nl,:) = center_finite_diff_n(abo_vor_clm(nl:nl,:), dx, True, 0, 1)
        end do

        do ml = 0, mlon-1  
            dy = 6378388. * dlat 
            dvordy(:,:,ml:ml) = center_finite_diff_n(abo_vor(:,:,ml:ml), dy, False, 0, 1)
            dvordy_clm(:,ml:ml) = center_finite_diff_n(abo_vor_clm(:,ml:ml), dy, False, 0, 0)
        end do

        ;>---------------------------------------------------------------------------------------<
        ;>---------------------------------------------------------------------------------------<

        uvdu_clm := conform(uvdu, uvdu_clm, (/1, 2/))
        uvdv_clm := conform(uvdv, uvdv_clm, (/1, 2/))
        abo_vor_clm := conform(abo_vor, abo_vor_clm, (/1, 2/))
        dvordx_clm := conform(dvordx, dvordx_clm, (/1, 2/))
        dvordy_clm := conform(dvordy, dvordy_clm, (/1, 2/))
        dv_d_clm := conform(dvordy, dv_d_clm, (/1, 2/))

        S1 = - abo_vor_clm * dv_d
        S2 = - uvdu * dvordx_clm - uvdv * dvordy_clm
        S3 = - abo_vor * dv_d_clm
        S4 = - uvdu_clm * dvordx - uvdv_clm * dvordy

        S1 = S1 * 10^11
        S2 = S2 * 10^11
        S3 = S3 * 10^11
        S4 = S4 * 10^11

        RWS = S1 + S2 + S3 + S4

        copy_VarCoords(u_anom, S1)
        copy_VarCoords(u_anom, S2)
        copy_VarCoords(u_anom, S3)
        copy_VarCoords(u_anom, S4)
        copy_VarCoords(u_anom, RWS)

        return [/RWS, S1, S2, S3, S4/]
    end

;;;>-----------------------------------------------------------------------------------------<

begin

;;;>-----------------------------------------------------------------------------------------<
;;;                                时间段
;;;>-----------------------------------------------------------------------------------------<

yyyymm_s_p1 = 197901
yyyymm_e_p1 = 199112

yyyymm_s_p2 = 199201
yyyymm_e_p2 = 200412

yyyymm_s_p3 = 200501
yyyymm_e_p3 = 202112

;;;   是否对环流场进行去年代际信号处理
rm_id_circ = False

data = 2   ;;;   1 JRA55 ;  2 ERA5

;;;>-----------------------------------------------------------------------------------------<
;;;                               读取数据
;;;>-----------------------------------------------------------------------------------------<

f_pc = addfile("./olr_index.nc", "r")
pc_p1 = f_pc->pc(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979)
pc_p2 = f_pc->pc(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979)
pc_p3 = f_pc->pc(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979)

pc_p1 = pc_p1 * (-1)
pc_p1 = dim_standardize(pc_p1, 1)

pc_p2 = pc_p2 * (-1)
pc_p2 = dim_standardize(pc_p2, 1)

pc_p3 = pc_p3 * (-1)
pc_p3 = dim_standardize(pc_p3, 1)

;;;>-----------------------------------------------------------------------------------------<

yyyymm_s = 197901
yyyymm_e = 202112

if (data .eq. 1) then

    idir = "./JRA55/mon/"

    f1 = addfile(idir + "pressure/anl_p125.007_hgt.195801_202312.nc", "r") 
    time = cd_calendar(f1->initial_time0_hours, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    hgt_200 = f1->HGT_GDS0_ISBL_S123(ind_s:ind_e,{250},:,:)

    dims_uvz = dimsizes(hgt_200)
    delete([/time, ind_s, ind_e, f1/])

    f2 = addfile(idir + "pressure/anl_p125.033_ugrd.195801_202312.nc", "r")
    time = cd_calendar(f2->initial_time0_hours, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    u_200 = f2->UGRD_GDS0_ISBL_S123(ind_s:ind_e,{250},:,:) 
    delete([/time, ind_s, ind_e, f2/])

    f3 = addfile(idir + "pressure/anl_p125.034_vgrd.195801_202312.nc", "r")  
    time = cd_calendar(f3->initial_time0_hours, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    v_200 = f3->VGRD_GDS0_ISBL_S123(ind_s:ind_e,{250},:,:) 
    delete([/time, ind_s, ind_e, f3/])

else

    idir = "./era5/monthly/"

    f1 = addfile(idir + "ERA5_Z_1x1_1940_2023_mon.nc", "r") 
    time = cd_calendar(f1->time, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    hgt_200 = short2flt(f1->z(ind_s:ind_e,{250},:,:))
    hgt_200 = hgt_200/9.8
    dims_uvz = dimsizes(hgt_200)
    delete([/time, ind_s, ind_e, f1/])

    f2 = addfile(idir + "ERA5_U_1x1_1940_2023_mon.nc", "r")
    time = cd_calendar(f2->time, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    u_200 = short2flt(f2->u(ind_s:ind_e,{250},:,:)) 
    delete([/time, ind_s, ind_e, f2/])

    f3 = addfile(idir + "ERA5_V_1x1_1940_2023_mon.nc", "r")  
    time = cd_calendar(f3->time, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    v_200 = short2flt(f3->v(ind_s:ind_e,{250},:,:)) 
    delete([/time, ind_s, ind_e, f3/])


end if

;;;-----------------------------------------------------------------------------------------------
;;;>-----------------------------------------------------------------------------------------<

printVarSummary(u_200)
; lat = u_200&g0_lat_2
lat = u_200&latitude
; lat = u_200&lat

if (lat(0) .gt. lat(1)) then
    u_200 := u_200(:,::-1,:)
    v_200 := v_200(:,::-1,:)
    hgt_200 := hgt_200(:,::-1,:)
    ; lat := u_200&g0_lat_2
    lat := u_200&latitude
    ; lat := u_200&lat
end if

;;;>-----------------------------------------------------------------------------------------<
;;;                                  季节平均 距平
;;;>-----------------------------------------------------------------------------------------<

n_year = dims_uvz(0)/12

hgt_200_anom = to_season(hgt_200)
u_200_anom = to_season(u_200)
v_200_anom = to_season(v_200)

u_200_clim = to_clim(u_200)
v_200_clim = to_clim(v_200)

delete([/hgt_200, u_200, v_200/])

;>---------------------------------------------------------------------------------------<
;;                    RWS
;>---------------------------------------------------------------------------------------<

temp = cal_rws(u_200_anom(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:), \
               v_200_anom(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:))
rws_p1 = temp[0]
delete(temp)

temp = cal_rws(u_200_anom(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:), \
               v_200_anom(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:))
rws_p2 = temp[0]
delete(temp)

temp = cal_rws(u_200_anom(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:), \
               v_200_anom(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:))
rws_p3 = temp[0]
delete(temp)

;;;>-----------------------------------------------------------------------------------------<
;;;                                      回归
;;;>-----------------------------------------------------------------------------------------<

hgt_200_r_p1 = regCoef_n(pc_p1, hgt_200_anom(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:), 0, 0)    
hgt_200_r_p2 = regCoef_n(pc_p2, hgt_200_anom(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:), 0, 0)    
hgt_200_r_p3 = regCoef_n(pc_p3, hgt_200_anom(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:), 0, 0)    
u_200_r_p1 = regCoef_n(pc_p1, u_200_anom(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:), 0, 0)    
u_200_r_p2 = regCoef_n(pc_p2, u_200_anom(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:), 0, 0)    
u_200_r_p3 = regCoef_n(pc_p3, u_200_anom(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:), 0, 0)   
v_200_r_p1 = regCoef_n(pc_p1, v_200_anom(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:), 0, 0)    
v_200_r_p2 = regCoef_n(pc_p2, v_200_anom(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:), 0, 0)    
v_200_r_p3 = regCoef_n(pc_p3, v_200_anom(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:), 0, 0)   

rws_r_p1 = regCoef_n(pc_p1, rws_p1, 0, 0)    
rws_r_p2 = regCoef_n(pc_p2, rws_p2, 0, 0)    
rws_r_p3 = regCoef_n(pc_p3, rws_p3, 0, 0)    

copy_VarCoords(hgt_200_anom(0,:,:), hgt_200_r_p1)
copy_VarCoords(hgt_200_anom(0,:,:), hgt_200_r_p2)
copy_VarCoords(hgt_200_anom(0,:,:), hgt_200_r_p3)
copy_VarCoords(hgt_200_anom(0,:,:), u_200_r_p1)
copy_VarCoords(hgt_200_anom(0,:,:), u_200_r_p2)
copy_VarCoords(hgt_200_anom(0,:,:), u_200_r_p3)
copy_VarCoords(hgt_200_anom(0,:,:), v_200_r_p1)
copy_VarCoords(hgt_200_anom(0,:,:), v_200_r_p2)
copy_VarCoords(hgt_200_anom(0,:,:), v_200_r_p3)
copy_VarCoords(hgt_200_anom(0,:,:), rws_r_p1)
copy_VarCoords(hgt_200_anom(0,:,:), rws_r_p2)
copy_VarCoords(hgt_200_anom(0,:,:), rws_r_p3)

;>---------------------------------------------------------------------------------------<
;;                    t-test
;>---------------------------------------------------------------------------------------<

confi_h200_p1 = confi(hgt_200_r_p1)
confi_h200_p2 = confi(hgt_200_r_p2)
confi_h200_p3 = confi(hgt_200_r_p3)

confi_rws_p1 = confi(rws_r_p1)
confi_rws_p2 = confi(rws_r_p2)
confi_rws_p3 = confi(rws_r_p3)

;>---------------------------------------------------------------------------------------<
;;                    WAF
;>---------------------------------------------------------------------------------------<

tn_p1 = tnwaf(hgt_200_r_p1, u_200_clim, v_200_clim)
fx_p1 = tn_p1[0]
fy_p1 = tn_p1[1]

tn_p2 = tnwaf(hgt_200_r_p2, u_200_clim, v_200_clim)
fx_p2 = tn_p2[0]
fy_p2 = tn_p2[1]

tn_p3 = tnwaf(hgt_200_r_p3, u_200_clim, v_200_clim)
fx_p3 = tn_p3[0]
fy_p3 = tn_p3[1]


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

res_sig = res
res_tn = res

res@vpWidthF = 0.8
res@vpHeightF = 0.4
res@mpShapeMode = "FreeAspect"
res@mpMinLatF = 00
res@mpMaxLatF = 90
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
res@tmXBLabelFontHeightF = 0.025
res@tmYLLabelFontHeightF = 0.025
res@tmXBTickSpacingF = 45
res@tmYLTickSpacingF = 20
res@tmXBMajorThicknessF = 1.5
res@tmYLMajorThicknessF = 1.5
res@tmXTOn = False
res@tmYROn = False

resh = res
res_rws = res

resh@cnLinesOn = False
resh@cnLineLabelsOn = False
resh@cnInfoLabelOn = False
resh@cnFillOn = True
resh@cnLevelSelectionMode = "ExplicitLevels"
resh@cnFillPalette = "MPL_BrBG"
resh@cnFillColors = (/100,95,90,85,80,-1,-1,45,40,35,30,25/)

resh@lbLabelBarOn = True
resh@lbOrientation = "Vertical"
resh@pmLabelBarWidthF = 0.05
resh@lbLabelFontHeightF = 0.02
resh@lbBoxEndCapStyle = "TriangleBothEnds"

res_rws@cnLinesOn = False
res_rws@cnLineLabelsOn = False
res_rws@cnInfoLabelOn = False
res_rws@cnFillOn = True
res_rws@lbLabelBarOn = True
res_rws@lbOrientation = "Vertical"
res_rws@pmLabelBarWidthF = 0.05
res_rws@lbLabelFontHeightF = 0.02
res_rws@lbBoxEndCapStyle = "TriangleBothEnds"   
res_rws@cnFillPalette = "BlWhRe"
res_rws@cnFillOpacityF = 0.5
res_rws@cnLevelSelectionMode = "ExplicitLevels"
res_rws@cnLevels = (/-10,-8,-6,-4,-2,-1,0,1,2,4,6,8,10/)
res_rws@cnFillColors = (/27,30,33,36,39,42,-1,-1,58,61,66,70,80,90/)

res_sig@cnFillOn = True
res_sig@lbLabelBarOn = False
res_sig@cnLinesOn = False
res_sig@cnInfoLabelOn = False
res_sig@cnLineLabelsOn = False
res_sig@cnMonoFillPattern = False

res_sig@cnLevelSelectionMode = "ExplicitLevels"
res_sig@cnLevels = (/0.1/)
res_sig@cnFillPatterns = (/17, -1/)
res_sig@cnFillColors = (/1, 0/)
res_sig@cnFillDrawOrder = "PostDraw"

res_tn@vcGlyphStyle = "CurlyVector"
res_tn@vcMinDistanceF = 0.023
res_tn@vcLineArrowThicknessF = 0.8
res_tn@vcLineArrowColor = "purple"
res_tn@vcLineArrowHeadMaxSizeF = 0.01
res_tn@vcLineArrowHeadMinSizeF = 0.01


res_tn@vcRefAnnoOn = True
res_tn@vcRefLengthF = 0.035
res_tn@vcRefAnnoBackgroundColor = "white"
res_tn@vcRefAnnoPerimOn = True   ;关闭边框
res_tn@vcRefAnnoFontHeightF = 0.008
res_tn@vcRefAnnoString1On =   False ;设置标注箭头上方不显示注释
res_tn@vcRefAnnoString2On = True   ;设置标注箭头下方显示注释
res_tn@vcRefAnnoFontThicknessF = 1.5

res_tn@vcRefAnnoOrthogonalPosF = -0.137
res_tn@vcRefAnnoParallelPosF = 0.08
res_tn@vcVectorDrawOrder = "PostDraw"

;;;>-----------------------------------------------------------------------------------------<

gres = True
gres@gsLineColor = "blue"
gres@gsLineThicknessF = 2

lonw1 = 130
lone1 = 210
lats1 = 5
latn1 = 22
x1 = (/lonw1,lone1,lone1,lonw1,lonw1/)
y1 = (/lats1,lats1,latn1,latn1,lats1/)

;;;>-----------------------------------------------------------------------------------------<

wks = gsn_open_wks("pdf", "./figs6")
    
;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

resh@gsnLeftString = "(a) Reg_Z250&WAF onto IPDC"
resh@gsnRightString = "P1"
resh@cnLevels := (/-15,-12,-9,-6,-3,0,3,6,9,12,15/)
; res@cnLevels := (/-20,-15,-10,-5,5,10,15,20/)
plot_h200_p1 = gsn_csm_contour_map(wks, hgt_200_r_p1, resh)
plot_h200_sig_p1 = gsn_csm_contour(wks, confi_h200_p1, res_sig)
overlay(plot_h200_p1, plot_h200_sig_p1)

resh@gsnLeftString = "(c) Reg_Z250&WAF onto IPDC"
resh@gsnRightString = "P2"
plot_h200_p2 = gsn_csm_contour_map(wks, hgt_200_r_p2, resh)
plot_h200_sig_p2 = gsn_csm_contour(wks, confi_h200_p2, res_sig)
overlay(plot_h200_p2, plot_h200_sig_p2)

resh@gsnLeftString = "(e) Reg_Z250&WAF onto IPDC"
resh@gsnRightString = "P3"
plot_h200_p3 = gsn_csm_contour_map(wks, hgt_200_r_p3, resh)
plot_h200_sig_p3 = gsn_csm_contour(wks, confi_h200_p3, res_sig)
overlay(plot_h200_p3, plot_h200_sig_p3)

;;;>-----------------------------------------------------------------------------------------<

res_tn@vcRefMagnitudeF = 0.1
res_tn@vcRefAnnoString2 = "0.1 m~S~2~N~/s~S~2~N~"
minf = 0.03 

fx_p1 = where(sqrt((/fx_p1/)^2 + (/fy_p1/)^2) .gt. minf, fx_p1, 0)
fy_p1 = where(sqrt((/fx_p1/)^2 + (/fy_p1/)^2) .gt. minf, fy_p1, 0)

fx_p2 = where(sqrt((/fx_p2/)^2 + (/fy_p2/)^2) .gt. minf, fx_p2, 0)
fy_p2 = where(sqrt((/fx_p2/)^2 + (/fy_p2/)^2) .gt. minf, fy_p2, 0)

fx_p3 = where(sqrt((/fx_p3/)^2 + (/fy_p3/)^2) .gt. minf, fx_p3, 0)
fy_p3 = where(sqrt((/fx_p3/)^2 + (/fy_p3/)^2) .gt. minf, fy_p3, 0)

plot_tn_p1 = gsn_csm_vector(wks, fx_p1, fy_p1, res_tn)
overlay(plot_h200_p1, plot_tn_p1)

plot_tn_p2 = gsn_csm_vector(wks, fx_p2, fy_p2, res_tn)
overlay(plot_h200_p2, plot_tn_p2)

plot_tn_p3 = gsn_csm_vector(wks, fx_p3, fy_p3, res_tn)
overlay(plot_h200_p3, plot_tn_p3)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

res_rws@gsnLeftString = "(b) Reg_RWS250 onto IPDC"
res_rws@gsnRightString = "P1"
plot_rws_p1 = gsn_csm_contour_map(wks, rws_r_p1, res_rws)
plot_rws_p1_sig = gsn_csm_contour(wks, confi_rws_p1, res_sig)
overlay(plot_rws_p1, plot_rws_p1_sig)

res_rws@gsnLeftString = "(d) Reg_RWS250 onto IPDC"
res_rws@gsnRightString = "P2"
plot_rws_p2 = gsn_csm_contour_map(wks, rws_r_p2, res_rws)
plot_rws_p2_sig = gsn_csm_contour(wks, confi_rws_p2, res_sig)
overlay(plot_rws_p2, plot_rws_p2_sig)

res_rws@gsnLeftString = "(f) Reg_RWS250 onto IPDC"
res_rws@gsnRightString = "P3"
plot_rws_p3 = gsn_csm_contour_map(wks, rws_r_p3, res_rws)
plot_rws_p3_sig = gsn_csm_contour(wks, confi_rws_p3, res_sig)
overlay(plot_rws_p3, plot_rws_p3_sig)

;;;>-----------------------------------------------------------------------------------------<

dum1 = gsn_add_polyline(wks, plot_h200_p1, x1, y1, gres)
dum2 = gsn_add_polyline(wks, plot_h200_p2, x1, y1, gres)
dum3 = gsn_add_polyline(wks, plot_h200_p3, x1, y1, gres)
dum4 = gsn_add_polyline(wks, plot_rws_p1, x1, y1, gres)
dum5 = gsn_add_polyline(wks, plot_rws_p2, x1, y1, gres)
dum6 = gsn_add_polyline(wks, plot_rws_p3, x1, y1, gres)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

pres = True
pres@gsnPanelXWhiteSpacePercent = 2
pres@gsnPanelYWhiteSpacePercent = 3

gsn_panel(wks, (/plot_h200_p1, plot_rws_p1 , \
                plot_h200_p2, plot_rws_p2, \
                plot_h200_p3, plot_rws_p3/), (/3, 2/), pres)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

end
