regression_timeseries_quantile_ar2_simu <- function(
    alpha1 = 0.5,
    alpha2 = 0.1,
    sim_len = 1000,
    tau=0.5,
    innovation_dis = "t",
    J_n = 9
){

# -------------------------------------------------------------------------------
# Simulate data from a nonparametric regression model with AR(2) errors and
# estimate the underlying smooth regression function and AR(2) parameters
# using quantile estimation.
#
# Arguments:
#   alpha1         AR(2) coefficient for lag 1 (default: 0.5).
#   alpha2         AR(2) coefficient for lag 2 (default: 0.1).
#   sim_len        Length of the simulated time series (default: 1000).
#   tau            Quantile level (e.g., 0.25, 0.50, or 0.75).
#   innovation_dis Innovation distribution. Available options are:
#                    "double_exponential"
#                    "t"
#                    "normal"
#                    "chisq5"
#                    "chisq15"
#                  Default: "t".
#   J_n            Number of basis functions (default: 9).
#
# Example:
#   regression_timeseries_quantile_ar2_simu(
#       alpha1 = 0.5,
#       alpha2 = 0.1,
#       sim_len = 1000,
#       tau = 0.5,
#       innovation_dis = "t",
#       J_n = 9
#   )
# -------------------------------------------------------------------------------


library(reticulate)
use_condaenv("r-reticulate", required = TRUE)
source_python('../quantile_ar2.py')
source_python('../leastsq_ar2.py')

set.seed(123456)


FNC = c('F1', 'F2', 'F3')

T_LEN = c(200, 400, 1000)


for (f_id in 1:length(FNC)) { 

for (l_id in 1:length(T_LEN)){


# simulate the data, AR(2) model

fct = FNC[f_id]
t_len = T_LEN[l_id]


ARP = 0.6   ## phi of AR(1), of the covariate variable process

for (i in 1:sim_len){


cat("smooth ", fct,  "AR2 simu", alpha1, " ", alpha2, "innovation ", innovation_dis, "sample path length ", t_len, "\n")


x_t=rnorm(t_len*1.2) # normal distribution


x_t=arima.sim(n=t_len, list(ar=c(ARP)), innov=x_t)  ## time series for the covariate variable  
rg=max(x_t)-min(x_t)
x_t=(x_t-min(x_t)+rg*0.005)/(max(x_t)-min(x_t)+rg*0.01)


if (fct=='F1') fx_t = 1-6*x_t+36*x_t^2 -53*x_t^3 +22*x_t^5
if (fct=='F2') fx_t = sin(x_t*2*pi)+2*x_t^2
if (fct=='F3') fx_t = atan((x_t-0.5)*5)-x_t^2/3 

if (innovation_dis=="t") inov = arima.sim(n=t_len*1.2 , model=list(ar = c(alpha1, alpha2)), rand.gen = function(n, ...) rt(n, df = 3))

if (innovation_dis=="chisq15") inov = arima.sim(n=t_len*1.2 , model=list(ar = c(alpha1, alpha2)), innov = rchisq(t_len*1.2, df=15) - 15)

if (innovation_dis=="chisq5") inov = arima.sim(n=t_len*1.2 , model=list(ar = c(alpha1, alpha2)), innov = rchisq(t_len*1.2, df=5) - 5)

if (innovation_dis=="normal") inov = arima.sim(n=t_len*1.2 , model=list(ar = c(alpha1, alpha2)), innov = rnorm(t_len*1.2))

if (innovation_dis=="double_exponential") inov = arima.sim(n=t_len*1.2 , model=list(ar = c(alpha1, alpha2)), innov = nimble::rdexp(t_len*1.2, location =0, scale = 1))

if (innovation_dis=="chisq15_variance_one") inov = arima.sim(n=t_len*1.2 , model=list(ar = c(alpha1, alpha2)), 
innov = (rchisq(t_len*1.2, df=15) - 15)/sqrt(30))

if (innovation_dis=="chisq5_variance_one") inov = arima.sim(n=t_len*1.2 , model=list(ar = c(alpha1, alpha2)), 
innov = (rchisq(t_len*1.2, df=5) - 5)/sqrt(10))


inov = inov[-(1:(t_len*0.2))]														
y_t = fx_t + inov


ts =y_t          # the vector of the response random variable, Y_t
xregr=x_t         # xregr is the regressor data.frame, currently we investigate the case that there is one regressor
n=length(ts)
nk=J_n+2
xregr<-as.matrix(xregr)	  # xregr is the regressor data.frame, currently we investigate the case that there is one regressor
SS = splines::bs(x=xregr, df = nk+2,  Boundary.knots=c(0,1), intercept = T)
ssknots = attr(SS, 'knots')

# least squares estimation
xreg <- SS[, -1] # drop the 'intercept'
param_start <- list(past_w=NULL, past_z=NULL, xregrc=NULL) 
param_start$xregrc=rep(1, dim(xreg)[2])   ################## initial values 
param_start$past_w=runif(2)               ################## initial values for AR(2)
	 while (any(abs( polyroot(c(1, -param_start$past_w)))<=1)){
      param_start$past_w=runif(2)
   }

starting_value <- unlist(param_start)
est_leastsq <- fit_leastsq_AR2(as.numeric(ts), xreg, starting_value)


cat('\n')
cat(i, ' least squares Python SLSQP estimates', est_leastsq[1:2], '\n')

# quantile estimation

xreg <- SS # keep the 'intercept'
param_start$xregrc=rep(1, dim(xreg)[2])   
starting_value <- unlist(param_start)

est_quantile <- fit_quantile_power_AR2(as.numeric(ts), xreg, starting_value, tau = tau)

cat('  quantile Python SLSQP estimates', est_quantile[1:2], '\n')

aaa= quantile_power_AR_ss2(as.numeric(ts), SS, est_quantile, tau = tau)

alpha_tau = aaa$alpha_tau
kappa = aaa$kappa

# estimation by arima()

fitSfx <- arima(ts, order=c(2,0, 0), xreg=SS[, -1], method='CSS')

cat(' ', 'arima() CSS estimates:', fitSfx$coef[1:2], '\n') 

fitSfx_default <- arima(ts, order=c(2,0,0), xreg=SS[, -1])

cat(' ', 'arima() default estimates:', fitSfx_default$coef[1:2], '\n') 

cat("smooth ", fct, 'tau ' , tau,  " AR2 simu", alpha1, " ", alpha2, "innovation ", innovation_dis, "length ", t_len, "\n")

cat("\n")
cat("\n")

# compare the estimated smooth funciton and the true smooth function

xx=seq(0, 1, len=2000)

if (fct=='F1') true_y =1-6*xx+36*xx^2 -53*xx^3 +22*xx^5

if (fct=='F2') true_y = sin(xx*2*pi)+2*xx^2

if (fct=='F3') true_y = atan((xx-0.5)*5)-xx^2/3


knots <-sort(c(rep(c(0,1), 4),  ssknots))
knots[4]=0
knots[length(knots)-3]=1
SL=splines::splineDesign(knots= knots, x = seq(0, 1, len=2000), ord=4)

SLQ = SL # for quantile
SL=SL[,-1]  # for least squares


fitted = SL%*%as.matrix(est_leastsq[3:length(est_leastsq)])+ mean(ts) - as.numeric(est_leastsq[3:length(est_leastsq)]%*%colMeans(SS[, -1])) 


qua_fitted = SLQ%*%as.matrix(est_quantile[3:(length(est_quantile))]-kappa)   


arima_fitted = SL%*%as.matrix(fitSfx$coef[4:length(fitSfx$coef)])+fitSfx$coef[3]

arima_default_fitted = SL%*%as.matrix(fitSfx_default$coef[4:length(fitSfx_default$coef)])+fitSfx_default$coef[3]


difnorm_least <- sum(((true_y-fitted)^2)[-1]*diff(xx))      # rho, least square method

difnorm_qua <- sum(((true_y-qua_fitted)^2)[-1]*diff(xx))    # rho, quantile median method

difnorm_arima <- sum(((true_y-arima_fitted)^2)[-1]*diff(xx)) # rho, arima() method, least squares


difnorm_arima_default <- sum(((true_y-arima_default_fitted)^2)[-1]*diff(xx)) #rho, arima() method, least squares


idx=xx>=0.1&xx<=0.9
true_y =true_y[idx]
fitted = fitted[idx]
qua_fitted=qua_fitted[idx]
arima_fitted = arima_fitted[idx]
arima_default_fitted = arima_default_fitted[idx]



difnorm_least19 <- sum(((true_y-fitted)^2)[-1]*diff(xx[idx]))  # rho19
difnorm_qua19 <- sum(((true_y-qua_fitted)^2)[-1]*diff(xx[idx])) # rho19
difnorm_arima19 = sum(((true_y-arima_fitted)^2)[-1]*diff(xx[idx])) # rho19
difnorm_arima_default19 = sum(((true_y-arima_default_fitted)^2)[-1]*diff(xx[idx])) # rho19

 
cat('\n', 'quan:      ',    'ls:       ',  'arima:css   ', 'arima:default  ',  
          'quan:     ' ,   'ls:      ',  'arima:css     ',  'arima:default  ',  '\n')


cat(  difnorm_qua, '  ', difnorm_least , ' ', difnorm_arima, ' ', difnorm_arima_default , 
' ' , difnorm_qua19, '  ', difnorm_least19, '  ', difnorm_arima19 ,
'  ', difnorm_arima_default19,  '\n')


# Store the simulation results in a .txt file using data.table::fwrite().
# The base R function write.table() can also be used as an alternative in this step.
data.table::fwrite( data.table::as.data.table(t(c(est_quantile[1:2], est_leastsq[1:2], fitSfx$coef[1:2], fitSfx_default$coef[1:2],t_len, 
    difnorm_qua,    difnorm_least,  difnorm_arima, difnorm_arima_default, 
		difnorm_qua19, difnorm_least19, difnorm_arima19, difnorm_arima_default19, alpha_tau, kappa, tau, J_n))), 
   file=paste0('quantile ', tau, ' ', innovation_dis,  ' innov AR2 ', fct, ' ', alpha1, ' ' , alpha2, ' ', t_len,  ' ', tau, '.txt'), append = T,col.names = F, 
row.names = F, sep=' ', buffMB = 4, nThread = 2)	

 }
}    # for t_len iteration
}    # for fct iteration

}