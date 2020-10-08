options(shiny.port = 8888)
# Load R packages
library(shiny)
library(shinythemes)
library(vroom)
library("e1071")
setwd("C:/Users/Admin/Documents/R/R Scripts/cab")
model=readRDS('final_model.rds')

# 1.Creating Distance feature in Dataset based on lat long positions
distance=function(dataset){
  deg_to_rad = function(deg){
    (deg * pi) / 180
  }
  haversine = function(long1,lat1,long2,lat2){
    #long1rad = deg_to_rad(long1)
    phi1 = deg_to_rad(lat1)
    #long2rad = deg_to_rad(long2)
    phi2 = deg_to_rad(lat2)
    delphi = deg_to_rad(lat2 - lat1)
    dellamda = deg_to_rad(long2 - long1)
    
    a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
      sin(dellamda/2) * sin(dellamda/2)
    
    c = 2 * atan2(sqrt(a),sqrt(1-a))
    R = 3959
    R * c #calculating in miles
  }
  
  dataset[,'distance']=NA
  dataset$distance = round(haversine(dataset$pickup_longitude,dataset$pickup_latitude,dataset$dropoff_longitude,dataset$dropoff_latitude),2)
  
  dataset=dataset[,-which(names(dataset) %in% c('pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'))]
  
  return(dataset)
}

# 2.Creating Time Based features in Dataset based on Pickup_datetime feature
time_features=function(dataset){
  dataset$pickup_datetime = strptime(as.character(dataset$pickup_datetime),"%Y-%m-%d %H:%M:%S")
  
  dataset$year = as.numeric(format(dataset$pickup_datetime,"%Y"))
  dataset$month = as.numeric(format(dataset$pickup_datetime,"%m"))
  dataset$week_day = as.numeric(format(dataset$pickup_datetime,"%w"))# Sunday = 0
  dataset$hour = as.numeric(format(dataset$pickup_datetime,"%H"))
  
  dataset=dataset[,-which(names(dataset) %in% c('pickup_datetime'))]
  
  return(dataset)
}

# 3.Creating Cab_type feature in Dataset based on passanger_count feature [small cab=0, bigger cab=1]
cab_type=function(dataset){
  dataset[,'cab_type']=0
  dataset$cab_type[which(dataset$passenger_count > 3)] = 1
  
  dataset=dataset[,-which(names(dataset) %in% c('passenger_count'))]
  
  return(dataset)
}

# 4.Creating scaling function
scaling=function(dataset){
  
  cnames = colnames(dataset)
  
  for(i in cnames){
    
    dataset[,i] = (dataset[,i] - mean(dataset[,i])) / sd(dataset[,i])
  }
  
  return(dataset)
}
#column_toDrop=readRDS('column_toDrop_list.rds')
# Define UI
ui <- fluidPage(theme = shinytheme("yeti"),
                navbarPage(
                   theme = "cerulean",  
                   tags$h3('Cab Fare Prediction '),
  tabPanel( 
            
           tags$h4('App'),
           tags$h5('This is an app to predict the fare of a particular ride, on the basis of pickup, dropoff location & Passenger count.'),
  )),              
  sidebarPanel(fileInput("file", 'Upload CSV File', accept = c(".csv"))),
  mainPanel(textOutput(outputId = "Prediction"),
            downloadButton("downloadData", "Download"),
            tableOutput("head"))
  
) # fluidPage


# Define server function  
server <- function(input, output, session) {
  
  
  df=data()
  data <- reactive({
    req(input$file)
    df <- read.csv(input$file$datapath)
    df_copy=df
    df<-distance(df)
    df<-time_features(df)
    df<-cab_type(df)
    df<-scaling(df)
    
    df_copy$fare_amount = predict(model, dat = as.matrix(df))
    
    df_copy
    
  })
  
  output$head <- renderTable(data())
  
  output$Prediction <- renderText('Predictions: ') 
  
  output$downloadData <- downloadHandler(
    filename = function() {
      paste("data-", "predictions", ".csv", sep="")
    },
    content = function(file) {
      write.csv(data(), file,row.names = FALSE)
    }
  )
  
}

# Create Shiny object
app <- shinyApp(ui = ui, server = server)
