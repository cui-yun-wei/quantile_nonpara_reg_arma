regression_timeseries_quantile_ma1_simu <-function(beta1=0.5,  sim_len = 1000, tau, innovation_dis = "t", J_n = 9){

# AR(1)


library(reticulate)
use_condaenv("r-reticulate", required = TRUE)
source_python('../quantile_ma1.py')
source_python('../leastsq_ma1.py')



set.seed(123456)



FNC = c('F1', 'F2', 'F3')

#FNC = c('F3')

#FNC = c('F1', 'F2')

T_LEN = c(200, 400, 1000)

#T_LEN = (1000)

for (f_id in 1:length(FNC)) { 

for (l_id in 1:length(T_LEN)){

#t_len= 2000          # length of the simulated data
# fct = 'F1'

fct = FNC[f_id]
t_len = T_LEN[l_id]


ARP = 0.6   ## phi of AR(1), of the covariate variable process


for (i in 1:sim_len){

cat(i, "\n")
cat(fct, " sample path length ", t_len, " ", innovation_dis, "\n")
cat("smooth ", fct,  "MA1 simu", beta1, " ", "quantile ", tau, "\n")


x_t=rnorm(t_len*1.2) # normal distribution


x_t=arima.sim(n=t_len, list(ar=c(ARP)), innov=x_t)  ## time series for the covariate variable  
rg=max(x_t)-min(x_t)
x_t=(x_t-min(x_t)+rg*0.005)/(max(x_t)-min(x_t)+rg*0.01)


if (fct=='F1') fx_t = 1-6*x_t+36*x_t^2 -53*x_t^3 +22*x_t^5
if (fct=='F2') fx_t = sin(x_t*2*pi)+2*x_t^2
if (fct=='F3') fx_t = atan((x_t-0.5)*5)-x_t^2/3 

if (innovation_dis=="t") inov = arima.sim(n=t_len*1.2 , model=list(ma = c(beta1)), rand.gen = function(n, ...) rt(n, df = 3))

if (innovation_dis=="chisq15") inov = arima.sim(n=t_len*1.2 , model=list(ma = c(beta1)), innov = rchisq(t_len*1.2, df=15) - 15)

if (innovation_dis=="chisq5") inov = arima.sim(n=t_len*1.2 , model=list(ma = c(beta1)), innov = rchisq(t_len*1.2, df=5) - 5)

if (innovation_dis=="normal") inov = arima.sim(n=t_len*1.2 , model=list(ma = c(beta1)), innov = rnorm(t_len*1.2))

if (innovation_dis=="double_exponential") inov = arima.sim(n=t_len*1.2 , model=list(ma = c(beta1)), innov = nimble::rdexp(t_len*1.2, location =0, scale = 1))

if (innovation_dis=="chisq15_variance_one") inov = arima.sim(n=t_len*1.2 , model=list(ma = c(beta1)), 
innov = (rchisq(t_len*1.2, df=15) - 15)/sqrt(30))

if (innovation_dis=="chisq5_variance_one") inov = arima.sim(n=t_len*1.2 , model=list(ma = c(beta1)), 
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


xreg <- SS[, -1] # drop the 'intercept'
param_start <- list(past_w=NULL, past_z=NULL, xregrc=NULL) 
param_start$xregrc=rep(1, dim(xreg)[2])   ################## initial values 
param_start$past_w=runif(1)         ############### initial values for AR(2)
	 
starting_value <- unlist(param_start)
est_leastsq <- fit_leastsq_MA1(as.numeric(ts), xreg, starting_value)


cat('\n')
cat(i, ' least squares Python SLSQP estimates', est_leastsq[1], '\n')
###########################################################################

xreg <- SS # keep the 'intercept'
param_start$xregrc=rep(1, dim(xreg)[2])   ################## only need change the basis coefficients for the initial values 

starting_value <- unlist(param_start)

est_quantile <- fit_quantile_MA1(as.numeric(ts), xreg, starting_value, tau = tau)

cat('XXX  quantile Python SLSQP estimates', est_quantile[1], '\n')

aaa= quantile_MA1_ss2(as.numeric(ts), SS, est_quantile, tau = tau)

alpha_tau = aaa$alpha_tau
kappa = aaa$kappa
################################################################################################

#fitSfx <- arima(ts, order=c(0,0,2), xreg=SS[, -1], method='CSS', optim.control=list(maxit=20000, reltol=1e-12))

fitSfx <- arima(ts, order=c(0 ,0, 1), xreg=SS[, -1], method='CSS')

cat(' ', 'arima() CSS estimates:', fitSfx$coef[1], '\n') 

fitSfx_default <- arima(ts, order=c(0,0,1), xreg=SS[, -1])

cat(' ', 'arima() default estimates:', fitSfx_default$coef[1], '\n') 

#cat("smooth ", fct, 'tau ' , tau,  " AR1 simu", beta1, " ", "innovation ", innovation_dis, "length ", t_len, "\n")



##################################################################################################

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


fitted = SL%*%as.matrix(est_leastsq[2:length(est_leastsq)])+ mean(ts) - as.numeric(est_leastsq[2:length(est_leastsq)]%*%colMeans(SS[, -1])) 


qua_fitted = SLQ%*%as.matrix(est_quantile[2:(length(est_quantile))]-kappa)  ### be careful with the sign of kappa 


arima_fitted = SL%*%as.matrix(fitSfx$coef[3:length(fitSfx$coef)])+fitSfx$coef[2]

arima_default_fitted = SL%*%as.matrix(fitSfx_default$coef[3:length(fitSfx_default$coef)])+fitSfx_default$coef[2]


difnorm_least <- sum(((true_y-fitted)^2)[-1]*diff(xx))      # rho, least square method

difnorm_qua <- sum(((true_y-qua_fitted)^2)[-1]*diff(xx)) #rho, quantile median method

difnorm_arima <- sum(((true_y-arima_fitted)^2)[-1]*diff(xx)) #rho, arima() method, least squares


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


cat("\n")
cat("\n")

data.table::fwrite( t(c(est_quantile[1], est_leastsq[1], fitSfx$coef[1], fitSfx_default$coef[1],t_len, 
    difnorm_qua,    difnorm_least,  difnorm_arima, difnorm_arima_default, 
		difnorm_qua19, difnorm_least19, difnorm_arima19, difnorm_arima_default19, alpha_tau, kappa, tau, J_n)), 
   file=paste0('quantile ', tau, ' ', innovation_dis,  ' innov MA1 ', fct, ' ', beta1, ' ', t_len, '.txt'), append = T,col.names = F, row.names = F, sep=' ')	

 }
}    # for t_len iteration
}    # for fct iteration


t500=read.table(paste0('quantile ', tau, ' ', innovation_dis,  ' innov MA1 ', fct, ' ', beta1, ' ', t_len, '.txt')) 
apply(t500[,1:4], 2, mean)   # mean of estimates               
apply(t500[, 1:4], 2, sd)   # sd of estimated
apply(t500[, 6:7], 2, mean) # mean of the norm between g and g_hat


}