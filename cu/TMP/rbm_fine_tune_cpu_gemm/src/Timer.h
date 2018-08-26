//
//  Timer.h
//  GaussianFilter
//
//  Created by hylo on 4/22/14.
//  Copyright (c) 2014 Parallel Software Design Lab, UOS. All rights reserved.
//

#ifndef HYLO_TIMER_H
#define HYLO_TIMER_H

#include <stdio.h>

const int MAX_NUM_TIMER = 100;
const int MAX_LEN_ALIAS = 100;



bool	initTimer();
void	startTimer(const char* timerName );
double	endTimer(const char* timerName );
double	endTimerp(const char* timerName );
double getTimer(const char* timerName );
void getTimerp(const char* timerName );
void	destroyTimer();
void setTimer(const char* timerName );
#endif