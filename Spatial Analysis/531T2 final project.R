####################
## PRE-PROCESSING ##
####################

#load libraries
lapply(c("ggmap","zoo","ggplot2","png","RJSONIO","RgoogleMaps","maps","fields","sm","spatstat","splancs","spatial","geoR","MASS","ggplot2",),require,character.only=TRUE)


#read in data
setwd("C:/Users/gjohnson/Desktop/FP")
ufo<-read.csv("ufofinal.csv",colClasses=c("factor","character","numeric","numeric"))

#format dates
ufo[["Date"]]<-as.Date(ufo[["Date"]],format="%m/%d/%Y")
ufo[["Year"]]<-as.numeric(format(ufo[["Date"]],'%Y'))


###############################
## EXPLORATORY DATA ANALYSIS ##
###############################

#as a monthly time series with data aggregated over the spatial dimension
month<-as.yearmon(ufo[["Date"]])
tmp<-ts(data=as.data.frame(table(month))[,2],start=c(2010,1),frequency=12)
plot(tmp,type="o",col="blue",lwd=2,frame.plot=FALSE,xlab="Year",ylab="Number of UFO Sightings")


#plot for all of the spatial data over the 6 years
zoom = 6
lat = c(min(ufo$lat),max(ufo$lat))
lon = c(min(ufo$lon),max(ufo$lon))
center = c(lat=mean(lat),lon=mean(lon))
MyMap1 = GetMap(center=center,zoom=zoom,maptype="mobile",GRAYSCALE="FALSE")
transform = LatLon2XY.centered(MyMap1,ufo$lat,ufo$lon)
PlotOnStaticMap(MyMap1)
points(transform$newX,transform$newY,col="darkred",pch=3)

#create a spatial plot for each year
for(year in 2010:2015){
  lat<-ufo[ufo$Year==year,"lat"]
  lon<-ufo[ufo$Year==year,"lon"]
  transform = LatLon2XY.centered(MyMap1,lat,lon)
  PlotOnStaticMap(MyMap1)
  points(transform$newX,transform$newY,col="darkred",pch=3)
}
#we can treat these as separate realizations - they look independent

#how about months as separate realizations?
#create a spatial plot for each month
for(mon in unique(month)){
  map("state",c("california"),xlim=c(-125,-113.5),ylim=c(32,43),lwd=2)
  with(ufo[month==mon,],points(lon,lat,col="darkred",pch=3))
}

#prototype over the months
proto<-read.csv("prototype.csv",sep=",") #I created the prototype on gf's computer since the package would only work on a Mac
transform = LatLon2XY.centered(MyMap1,proto[,1],proto[,2])
PlotOnStaticMap(MyMap1)
points(transform$newX,transform$newY,col="darkred",pch=3)


#################################
## POINT PROCESS DATA ANALYSIS ##
#################################

#remove multiple sightings in one city i.e. convert to point process

ufo_pp<-unique(ufo[,c("Date","lat","lon","Year")])

#Ripley's K and L function
x1<-ufo_pp[["lon"]]
y1<-ufo_pp[["lat"]]
n = length(x1)
bdw = sqrt(bw.nrd0(x1)^2+bw.nrd0(y1)^2)

b1 = as.points(x1,y1)
load("data/CA outline data coverage areas.Rdata")
bdry = ca.outline

s = seq(.001,.7,length=50)
k4 = khat(b1,bdry,s)
plot(s,k4,xlab="Distance",ylab="K4(h)",pch="*",xlim=c(0,.4),ylim=c(0,.05))
lines(s,k4)
lines(s,pi*s^2,lty=2)
L4 = sqrt(k4/pi)-s
plot(c(.001,.7),range(L4),type="n",xlab="lag, h",ylab="L4(h) - h")
points(s,L4,pch="*")
lines(s,L4)
lines(s,rep(0,50),lty=2)

###Simulated bounds for K
k4conf = Kenv.csr(npts(b1), bdry, 100, s)
plot(c(0,max(s)),c(0,max(k4conf$upper,k4)),
     type="n",xlab="Distance",ylab="K4(h)",main="K4 Function")
points(s,k4,pch="*")
lines(s,k4)
lines(s,k4conf$upper,lty=3)
lines(s,k4conf$lower,lty=3)

L4upper = sqrt(k4conf$upper/pi) - s
L4lower = sqrt(k4conf$lower/pi) - s
plot(c(0,max(s)),c(min(L4lower,L4),max(L4upper,L4)),
     type="n",xlab="distance",ylab="L4(h) - h",main="L4 Function")
points(s,L4,pch="*")
lines(s,L4)
lines(s,L4upper,lty=2)
lines(s,L4lower,lty=2)

###Kernel density estimation
ufo.density<-kde2d(ufo$lon,ufo$lat,h=.7,n=200)

ggplot(data=ufo_pp,aes(lon,lat))+
  stat_density2d(aes(alpha=..level..), geom="polygon",h=.8) +
  scale_alpha_continuous(limits=c(0,0.2),breaks=seq(0,0.2,by=0.025))+
  geom_point(colour="red",alpha=0.2)+
  theme_bw()

ggplot(data=ufo_pp,aes(lon,lat)) + 
  stat_density2d(aes(fill=..level..,alpha=..level..),geom='polygon',colour='black',h=.8) + 
  scale_fill_continuous(low="green",high="red") +
  #geom_polygon(data=as.data.frame(ca.outline),mapping=aes(V1,V2),fill=NA,col="black",size=1) +
  guides(alpha="none") +
  geom_point() + 
  list(labs(color="Density",fill="Density",
                        x="Longitude",
                        y="Latitude"),
                   theme_bw(),
                   theme(legend.position=c(0,1),
                         legend.justification=c(0,1)))
library(RColorBrewer)
rf <- colorRampPalette(rev(brewer.pal(11,'Spectral')))
r <- rf(32)
require(hexbin)
h<-hexbin(ufo_pp[,c("lon","lat")],xbins=50)
plot(h,colramp=rf)
