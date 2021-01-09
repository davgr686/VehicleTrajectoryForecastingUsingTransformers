library("ggplot2")
library("reshape2")

data = read.csv2("tdata.csv", header = TRUE)

trajdf <- data.frame(
  time = data$time,
  in.X = data$input.X,
  in.Y = data$input.Y,
  pred.X = data$pred.X,
  pred.Y = data$pred.Y,
  true.X = data$truth.x,
  true.Y = data$truth.Y
)


ggplot(trajdf, aes(x = in.X, y = in.Y, color = "Input Trajectory")) + geom_path(size = 3) +
  geom_path(trajdf, mapping = aes(x = pred.X, y = pred.Y, col = "Predicted Trajectory"), size = 1.5, linetype = "longdash") +
  geom_path(trajdf, mapping = aes(x = true.X, y = true.Y, col = "True Trajectory"), size = 1.5, linetype = "dashed") +
  xlim(15, 26) + ylim(15,25) + labs(title = "Trajectories", x = "X", y = "Y", color = "Legend") 
  

         