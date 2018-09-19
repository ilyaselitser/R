### https://archive.ics.uci.edu/ml/datasets/Wilt
library(rjags)
library(UBL)

dat = read.csv('training.csv')
head(dat)

dim(dat)
sum(dat$class == 'w')
sum(dat$class == 'n')

# Rename columns 
# wilt = 0 means class == ‘n’, wilt = 1 means class == ‘w’
colnames(dat) = c("wilt","glcm", "mgreen","mred","mnir","sdp")

#To correct the imbalance to 1:1 the SMOTE algorithm from UBL package was used
dat_wilt = SmoteClassif(wilt ~ . , dat,  C.perc = "balance", k = 5, repl = FALSE, dist = 'Euclidean', p = 2)
dat_wilt$wilt = as.numeric(dat_wilt$wilt)
dat_wilt$wilt[dat_wilt$wilt == 1] = 0
dat_wilt$wilt[dat_wilt$wilt == 2] = 1

sum(dat_wilt$wilt == 0)
sum(dat_wilt$wilt == 1)

pairs(dat_wilt)

# mgreen - mred relation
plot(dat_wilt$mgreen, dat_wilt$mred)

# Scale X
X = scale(dat_wilt[, 2:6], center=TRUE, scale=TRUE)
# Columns means should be  close to 0
colMeans(X)

# calculate standard deviation of X
apply(X, 2, sd)

mod1_string = " model {
for (i in 1:length(y)) {

y[i] ~ dbern(p[i])
logit(p[i]) = int + b[1]*glcm[i] + b[2]*mgreen[i] + b[3]*mred[i] + b[4]*mnir[i] + b[5]*sdp[i]
}
int ~ dnorm(0.0, 1.0/1e4)
for (j in 1:5) {
b[j] ~ ddexp(0.0, sqrt(2.0)) # has variance 1.0
}
} "

set.seed(118)

data_jags = list(y=dat_wilt$wilt, glcm=X[,"glcm"], mgreen=X[,"mgreen"], mred=X[,"mred"], mnir=X[,"mnir"], sdp=X[,"sdp"])

params = c("int", "b")

mod1 = jags.model(textConnection(mod1_string), data=data_jags, n.chains=3)

update(mod1, 1e4)

mod1_sim = coda.samples(model=mod1, variable.names=params, n.iter=5e4)
mod1_csim = as.mcmc(do.call(rbind, mod1_sim))
summary(mod1_csim)

## convergence diagnostics
plot(mod1_sim, ask=TRUE)

gelman.diag(mod1_sim)
autocorr.diag(mod1_sim)
autocorr.plot(mod1_sim)
effectiveSize(mod1_sim)

## calculate DIC
dic1 = dic.samples(mod1, n.iter=1e4)
par(mfrow=c(3,2))
densplot(mod1_csim[,1:5], xlim=c(-40.0, 40.0))

pm_coef1 = colMeans(mod1_csim)
pm_Xb1 = pm_coef1["int"] + X[, c(1:5)] %*%pm_coef1[1:5]
phat1 = 1.0/(1.0 + exp(-pm_Xb1))
plot(phat1, dat_wilt$wilt)

tab1_0.5 = table(phat1 > 0.5, dat_wilt$wilt)
sum(diag(tab1_0.5))/sum(tab1_0.5)

test_wilt = read.csv('testing.csv')
colnames(test_wilt) = c("wilt","glcm", "mgreen","mred","mnir","sdp")

# design matrix
X_test1 = scale(test_wilt[, 2:6], center=TRUE, scale=TRUE)
pm_Xtb1 = pm_coef1["int"] + X_test1[, c(1:5)] %*%pm_coef1[1:5]
phat_t_1 = 1.0/(1.0 + exp(-pm_Xtb1))
plot(phat_t_1, test_wilt$wilt)

tab1_t_0.5 = table(phat_t_1 > 0.5, test_wilt$wilt)
sum(diag(tab1_t_0.5))/sum(tab1_t_0.5)

# noninformative prior for logistic regression
mod2_string = " model {
for (i in 1:length(y)) {
y[i] ~ dbern(p[i])
logit(p[i]) = int + b[1]*glcm[i] + b[2]*mgreen[i] + b[3]*mred[i] + b[4]*mnir[i] + b[5]*sdp[i]
}
int ~ dnorm(0.0, 1.0/1e2)
for (j in 1:5) {
b[j] ~ dnorm(0.0, 1.0/1e2)
}
} "

mod2 = jags.model(textConnection(mod2_string), data=data_jags, n.chains=3)

update(mod2, 1e4)
mod2_sim = coda.samples(model=mod2, variable.names=params, n.iter=5e4)
mod2_csim = as.mcmc(do.call(rbind, mod2_sim))

plot(mod2_sim, ask=TRUE)

gelman.diag(mod2_sim)
autocorr.diag(mod2_sim)
autocorr.plot(mod2_sim)
effectiveSize(mod2_sim)

dic2 = dic.samples(mod2, n.iter=1e3)
par(mfrow=c(3,2))
densplot(mod2_csim[,1:5], xlim=c(-40.0, 40.0))

pm_coef2 = colMeans(mod2_csim)


# design matrix
pm_Xb2 = pm_coef2["int"] + X[, c(1:5)] %*%pm_coef2[1:5]
phat2 = 1.0/(1.0 + exp(-pm_Xb2))
plot(phat2, dat_wilt$wilt)

tab2_0.5 = table(phat2 > 0.5, dat_wilt$wilt)
sum(diag(tab2_0.5))/sum(tab1_0.5)

X_test2 = scale(test_wilt[, 2:6], center=TRUE, scale=TRUE)
pm_Xtb2 = pm_coef2["int"] + X_test2[, c(1:5)] %*%pm_coef2[1:5]
phat_t_2 = 1.0/(1.0 + exp(-pm_Xtb2))
plot(phat_t_2, test_wilt$wilt)

tab2_t_0.5 = table(phat_t_2 > 0.5, test_wilt$wilt)
sum(diag(tab2_t_0.5))/sum(tab2_t_0.5)