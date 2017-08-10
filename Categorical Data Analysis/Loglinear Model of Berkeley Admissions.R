### Berkeley 
#############################

head(UCBAdmissions) 
berk.data<-as.data.frame(UCBAdmissions)
#1-A,2-S,3-D


##MODEL 1## - Full Independence
#loglinear model (A,S,D)
loglin1<-loglin(UCBAdmissions, list(1,2,3), fit=TRUE, param=TRUE)
#poisson glm (Y~A+S+D)
glm1<-glm(Freq~Admit+Gender+Dept,data=berk.data,family=poisson)


##MODEL 2## - Admit-Gender Interaction
#loglinear model (AS,D)
loglin2<-loglin(UCBAdmissions, list(c(1,2),3), fit=TRUE, param=TRUE)
#poisson glm (Y~A+S+D+A*S)
glm2<-glm(Freq~Admit*Gender+Dept,data=berk.data,family=poisson)

#MODEL 3## - Admit-Dept Interaction
#loglinear model (AD,S)
loglin3<-loglin(UCBAdmissions, list(c(1,3),2), fit=TRUE, param=TRUE)
#poisson glm (Y~A+S+D+A*D)
glm3<-glm(Freq~Admit*Dept+Gender,data=berk.data,family=poisson)

#MODEL 4## - Gender-Dept Interaction
#loglinear model (SD,A)
loglin4<-loglin(UCBAdmissions, list(c(2,3),1), fit=TRUE, param=TRUE)
#poisson glm (Y~A+S+D+S*D)
glm4<-glm(Freq~Admit+Dept*Gender,data=berk.data,family=poisson)

#Model 5 - (AD,AS)
loglin5<-loglin(UCBAdmissions, list(c(1,3),c(1,2)), fit=TRUE, param=TRUE)
glm5<-glm(Freq~Admit*Dept+Admit*Gender,data=berk.data,family=poisson)

#Model 6 - (AS,DS)
loglin6<-loglin(UCBAdmissions, list(c(1,2),c(2,3)), fit=TRUE, param=TRUE)
glm6<-glm(Freq~Admit*Gender+Dept*Gender,data=berk.data,family=poisson)

#Model 7 - (AD,DS)
loglin7<-loglin(UCBAdmissions, list(c(1,3),c(2,3)), fit=TRUE, param=TRUE)
glm7<-glm(Freq~Admit*Dept+Dept*Gender,data=berk.data,family=poisson)

#Model 8 - (AS,AD)
loglin8<-loglin(UCBAdmissions, list(c(1,3),c(1,3)), fit=TRUE, param=TRUE)
glm8<-glm(Freq~Admit*Gender+Dept*Admit,data=berk.data,family=poisson)

#model 9 - (AS,AD,DS)
loglin9<-loglin(UCBAdmissions, list(c(1,3),c(1,2),c(2,3)), fit=TRUE, param=TRUE)
glm9<-glm(Freq~Admit*Dept+Admit*Gender+Dept*Gender,data=berk.data,family=poisson)

#model 10 - (ASD)
loglin10<-loglin(UCBAdmissions, list(c(1,2,3)), fit=TRUE, param=TRUE)
glm10<-glm(Freq~Admit*Dept*Gender,data=berk.data,family=poisson)



####QUESTION 1
#Appropriateness ofmodels? G2, Pearson, X2

#Goodness of Fit - Pearson
loglin1[["pearson"]]
loglin2[["pearson"]]
loglin3[["pearson"]]
loglin4[["pearson"]]
loglin5[["pearson"]]
loglin6[["pearson"]]
loglin7[["pearson"]]    #2nd best
loglin8[["pearson"]]    #2nd best
loglin9[["pearson"]]    #best
loglin10[["pearson"]]

#Absolute fit - LRT
pchisq(loglin1[["lrt"]],loglin1[["df"]],lower.tail=FALSE)
pchisq(loglin2[["lrt"]],loglin2[["df"]],lower.tail=FALSE)
pchisq(loglin3[["lrt"]],loglin3[["df"]],lower.tail=FALSE)
pchisq(loglin4[["lrt"]],loglin4[["df"]],lower.tail=FALSE)
pchisq(loglin5[["lrt"]],loglin5[["df"]],lower.tail=FALSE)
pchisq(loglin6[["lrt"]],loglin6[["df"]],lower.tail=FALSE)
pchisq(loglin7[["lrt"]],loglin7[["df"]],lower.tail=FALSE)
pchisq(loglin8[["lrt"]],loglin8[["df"]],lower.tail=FALSE)
pchisq(loglin9[["lrt"]],loglin9[["df"]],lower.tail=FALSE)
pchisq(loglin10[["lrt"]],loglin10[["df"]],lower.tail=FALSE) #perfect fit!



####QUESTION 2
#Interpret admission vs. gender in Dept A vs. Dept F in all models


###Question 3
#Perform a model selection







### Via glm() function #######
berk.data<-as.data.frame(UCBAdmissions)
berk.data
berk.ind<-glm(berk.data$Freq~berk.data$Admit+berk.data$Gender+berk.data$Dept, family=poisson())
summary(berk.ind)
fits<-fitted(berk.ind)
resids <- residuals(berk.ind,type="pearson")
adjresids <- resids/sqrt(1-h)
round(cbind(berk.data$Freq,fits,adjresids),2)

### Via loglin() function
berk.ind<-loglin(UCBAdmissions, list(1,2,3), fit=TRUE, param=TRUE)
berk.ind

##### Saturated log-linear model 
### via loglin()
berk.sat<-loglin(UCBAdmissions, list(c(1,2,3)), fit=TRUE, param=TRUE)
berk.sat

### via glm()
berk.sat<-glm(berk.data$Freq~berk.data$Admit*berk.data$Gender*berk.data$Dept, family=poisson())
summary(berk.sat)
fitted(berk.sat)
