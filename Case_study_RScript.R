rm(list = ls()) # Removes all variables from environment
cat("\f")

# Library

library(funcharts)
library(fda)
library(tidyverse)
library(lubridate)
library(stringr)
library(FuncNN)
library("keras")
library("latex2exp")
library(ggpubr)

# Import HVA data set
load("HVAC_data_set.RData")

# Phase I 

# To get smooth profiles, we choose a B-spline basis system with 70 basis functions 
# and equally spaced knots estimated by solving a regularization problem with a 
# roughness penalty on the integrated-squared second derivative and smoothing parameter chosen through a generalized cross-validation

loglam <- seq(-4,0,0.25)
lambda_grid <- 10^loglam

mfdobj_phaseI <- HVAC_dataset %>% 
  filter(Train %in% c("Train 1", "Train 2", "Train 3", "Train 4")) %>%
  bind_rows(HVAC_dataset %>%
              filter(Train == "Train 5") %>%
              filter(Time <= "10-25")) %>% # to remove the PhaseII observations
  arrange(Train, VN, Time) %>%
  filter(Percent_distance >= 0.25) %>% # the first 25% of the traveled distance of each profile is discarded
  group_by(VN) %>% 
  mutate(Percent_distance = (Percent_distance - 0.25)/0.75) %>%
  ungroup() %>%
  mutate(VN = factor(VN)) %>% 
  get_mfd_df(domain = c(0, 1),
             arg = "Percent_distance",
             id = "VN",
             lambda_grid = lambda_grid, #seq(0.05, 0.2, length=5)
             variables = c("coach_OutdoorTemp",
                           "coach_SetPointTemp"),
             n_basis = 70) # 70

# Derivative of the target temperature profile

dev_setpoint <-  deriv.fd(mfdobj_phaseI[,"coach_SetPointTemp"])


# The scalar quality characteristic

scalar_train <- HVAC_dataset %>% 
  filter(Train %in% c("Train 1", "Train 2", "Train 3", "Train 4")) %>%
  bind_rows(HVAC_dataset %>%
              filter(Train == "Train 5") %>%
              filter(Time <= "10-25")) %>%
  filter(VN %in% unique(mfdobj_phaseI[["raw"]][["VN"]])) %>%
  arrange(Train, VN, Time) %>%
  filter(Percent_distance >= 0.25) %>%
  select(DeltaTemp,VN,n) %>%
  group_by(VN) %>%
  mutate(dev_deltaTemp =sqrt(sum(DeltaTemp^2))/n) %>% 
  summarise(min = min(dev_deltaTemp)) %>% pull()

# Plot of a random slice of 100 observations of the temperature profiles

set.seed(3) 
rows <- sample(1:dim(mfdobj_phaseI$coefs)[2], 100)

p1<- plot_mfd(mfdobj_phaseI[rows,"coach_OutdoorTemp"]) +
  geom_line(mfdobj_phaseI = mfdobj_phaseI[rows], lty = 2, type_mfd = "raw", col = "darkgreen") +
  xlab("Fraction of distance traveled") +
  ylab("Outdoor temperature") 
p1 

dev_setpoint_plot <- get_mfd_fd(dev_setpoint)

set.seed(3) # 36
rows <- sample(1:dim(mfdobj_phaseI$coefs)[2], 100)

p2<- plot_mfd(dev_setpoint_plot[rows,]) +
  geom_line(dev_setpoint_plot = dev_setpoint_plot[rows], lty = 2, type_mfd = "raw", col = "darkgreen") +
  xlab("Fraction of distance traveled") +
  ylab("Derivative of the target temperature") 
p2

p<- ggarrange(p1,p2, ncol = 2, nrow = 1)
p

# Split train/tuning set

voyage_id_PhaseI <- as.vector(unique( mfdobj_phaseI[["raw"]][["VN"]]))
set.seed(1)
voyage_id_train <- sample(voyage_id_PhaseI, size = length(voyage_id_PhaseI)/2,
                          replace = FALSE)

voyage_id_tun <- setdiff(voyage_id_PhaseI, voyage_id_train)

# Phase II data

mfdobj_test <- HVAC_dataset %>% 
  filter(Train == "Train 5") %>% 
  filter(Percent_distance >= 0.25) %>% # the first 25% of the traveled distance of each profile is discarded
  group_by(VN) %>% 
  mutate(Percent_distance = (Percent_distance - 0.25)/0.75) %>%
  ungroup() %>%
  mutate(VN = factor(VN)) %>% 
  get_mfd_df(domain = c(0, 1),
             arg = "Percent_distance",
             id = "VN",
             lambda_grid = lambda_grid, #seq(0.05, 0.2, length=5)
             variables = c("coach_OutdoorTemp",
                           "coach_SetPointTemp"),
             n_basis = 70) # 70

dev_setpoint_test <-  deriv.fd(mfdobj_test[,"coach_SetPointTemp"])

# Response variable

scalar_test <- HVAC_dataset %>% 
  filter(Train == "Train 5") %>%
  filter(Percent_distance >= 0.25) %>%
  group_by(VN) %>%
  mutate(dev_deltaTemp =sqrt(sum(DeltaTemp^2))/n) %>% 
  summarise(min = min(dev_deltaTemp)) %>% pull()


# FNNCC

# Setting up tensor for fFNN
n_basis <- 70
data_fnnI <- array(dim = c(n_basis, length(voyage_id_train), 2))
data_fnnI[,,1] <- mfdobj_phaseI$coefs[,voyage_id_train,1]
data_fnnI[,,2] <- drop(dev_setpoint[["coefs"]])[,voyage_id_train]

data_fnnI_tun = array(dim = c(n_basis, length(voyage_id_tun), 2))
data_fnnI_tun[,,1] <- mfdobj_phaseI$coefs[,voyage_id_tun,1]
data_fnnI_tun[,,2] <- drop(dev_setpoint[["coefs"]])[,voyage_id_tun]

data_fnnII = array(dim = c(n_basis, dim(mfdobj_test$coefs)[2], 2))
data_fnnII[,,1] <-  mfdobj_test$coefs[,,1]
data_fnnII[,,2] <- drop(dev_setpoint_test[["coefs"]])

# FNN training

seed <- 1
set.seed(seed)
tensorflow::set_random_seed(seed)
fnn_sim <- fnn.fit(resp = scalar_train[voyage_id_PhaseI %in% voyage_id_train], 
                   func_cov = data_fnnI, 
                   scalar_cov = NULL,
                   basis_choice = c("bspline"), 
                   num_basis = c(5), 
                   hidden_layers = 2, 
                   neurons_per_layer = c(8,8), # (16,16); (8,4)
                   activations_in_layers = c("relu", "linear"), 
                   domain_range = list(c(0, 1)),
                   epochs = 250,
                   loss_choice = "mse",
                   metric_choice = list("mean_squared_error"),
                   val_split = 0.2,
                   patience_param = 25,
                   learn_rate = 0.001,
                   early_stop = T)

predictions <- fnn.predict(model = fnn_sim,
                           func_cov = data_fnnI_tun,
                           scalar_cov = NULL,
                           basis_choice = c("bspline"), 
                           num_basis = c(5),
                           domain_range = list(c(0, 1))
)

# Control limit estimation

res_tun <- scalar_train[voyage_id_PhaseI %in% voyage_id_tun] - predictions

UCL_FNN_np <- quantile(res_tun, probs = 1 - 0.05/2)
UCL_FNN_np
LCL_FNN_np <- quantile(res_tun, probs = 0.05/2)
LCL_FNN_np


predictionsII <- fnn.predict(model = fnn_sim,
                             func_cov = data_fnnII,
                             scalar_cov = NULL,
                             basis_choice = c("bspline"), 
                             num_basis = c(5),
                             domain_range = list(c(0, 1))
)

res <-  scalar_test - predictionsII # residuals

# Plot control chart

df_test <- cbind(res[rows], seq(1, length(res[rows])), 
                 res[rows] > UCL_FNN_np | res[rows] < LCL_FNN_np)  
df_test  <- as.data.frame(df_test)
names(df_test) <- c("res", "id", "oc")


FNNCC_plot <- ggplot(df_test, aes(x = id, y = res)) + 
  geom_line() + 
  geom_point(aes(colour = oc)) + 
  geom_blank(aes(y = 0)) + geom_line(aes(y = UCL_FNN_np), 
                                     lty = 2) + geom_line(aes(y = LCL_FNN_np), lty = 2) + 
  theme_bw() + theme(axis.text.x = element_text(angle = 90, 
                                                vjust = 0.5)) + theme(legend.position = "none", 
                                                                      plot.title = element_text(hjust = 0.5)) +
  xlab("Voyage index") + ylab(expression("FNN residuals")) 
#ggtitle("FNNCC")

FNNCC_plot + scale_x_continuous(breaks = seq(1, nrow(df_test), by = 2))
