//
//  Timer.cpp
//
//  Created by hylo on 4/22/14.
//  Copyright (c) 2014 Parallel Software Design Lab, UOS. All rights reserved.
//

#include "Timer.h"
#include "string.h"

#ifdef WIN32
#include <Windows.h>
LARGE_INTEGER li_st[MAX_NUM_TIMER];
LARGE_INTEGER li_ed[MAX_NUM_TIMER];
double PCFreq = 0.0;
#else
#include <sys/time.h>
struct timeval tv_st[MAX_NUM_TIMER], tv_ed[MAX_NUM_TIMER], tv_diff[MAX_NUM_TIMER];
#endif

double t[MAX_NUM_TIMER] = {0,};
int cnt[MAX_NUM_TIMER] = {0,};
int g_num_timer = 0;
const char* timer_alias[MAX_NUM_TIMER];


bool initTimer()
{
    g_num_timer = 0;
#ifdef WIN32
    if(!QueryPerformanceFrequency(&li_st[0])) {
	printf("QueryPerformanceFrequency Failed!\n");
	return false;
    }
    PCFreq = (double)li_st[0].QuadPart/1000.0;
#else

#endif

    return true;
}


void startTimer(const char* timerName )
{
    // Check already exist timer
    int i;
    for( i = 0 ; i < g_num_timer; ++i ) {
	if( !strcmp(timer_alias[i], timerName) )
	    break;
    }

    if( g_num_timer >= MAX_NUM_TIMER) {
	printf("Exceeded maximum number of timers\n");
	return;

    } else if( i == g_num_timer ) {
	timer_alias[g_num_timer] = timerName;
	g_num_timer++;
    }

    int target = i;
#ifdef WIN32
    QueryPerformanceCounter(&li_st[target]);
#else
    gettimeofday(&tv_st[target], NULL);
#endif

    cnt[target]++;
}


double endTimer(const char* timerName )
{
    // Find exist timer
    int i;
    for( i = 0 ; i < g_num_timer; ++i ) {
	if( !strcmp(timer_alias[i], timerName) )
	    break;
    }

    if( i == g_num_timer ) {
	printf("error: cannot find the timer \"%s\"\n", timerName);
	return -1;
    }

    int target = i;
#ifdef WIN32
    QueryPerformanceCounter(&li_ed[target]);
    t[target] += (double)(li_ed[target].QuadPart - li_st[target].QuadPart)/PCFreq;
#else
    gettimeofday(&tv_ed[target], NULL);
    timersub(&tv_ed[target], &tv_st[target], &tv_diff[target]);
    t[target] += tv_diff[target].tv_sec * 1000.0 + tv_diff[target].tv_usec/1000.0;
#endif

    return t[target];
}


double endTimerp(const char* timerName )
{
    // Find exist timer
    int i;
    static double prev = 0;

    for( i = 0 ; i < g_num_timer; ++i ) {
	if( !strcmp(timer_alias[i], timerName) )
	    break;
    }

    if( i == g_num_timer ) {
	printf("error: cannot find the timer \"%s\"\n", timerName);
	return -1;
    }

    int target = i;
#ifdef WIN32
    QueryPerformanceCounter(&li_ed[target]);
    t[target] += (double)(li_ed[target].QuadPart - li_st[target].QuadPart)/PCFreq;
#else
    gettimeofday(&tv_ed[target], NULL);
    timersub(&tv_ed[target], &tv_st[target], &tv_diff[target]);
    t[target] += prev = tv_diff[target].tv_sec * 1000.0 + tv_diff[target].tv_usec/1000.0;
#endif

    printf("[Timer] %s: %.2lf ms, count=%d, avg = %.2lf\n", timerName, t[target], cnt[target], t[target]/cnt[target]);
    //	return t[target] - prev;
    return t[target];
}

double getTimer(const char* timerName )
{
    // Find exist timer
    int i;

    for( i = 0 ; i < g_num_timer; ++i ) {
	if( !strcmp(timer_alias[i], timerName) )
	    break;
    }

    if( i == g_num_timer ) {
	printf("error: cannot find the timer \"%s\"\n", timerName);
	return -1;
    }

    int target = i;

    return t[target];
}

void getTimerp(const char* timerName )
{
    // Find exist timer
    int i;

    for( i = 0 ; i < g_num_timer; ++i ) {
	if( !strcmp(timer_alias[i], timerName) )
	    break;
    }

    if( i == g_num_timer ) {
	printf("error: cannot find the timer \"%s\"\n", timerName);
	return;
    }

    int target = i;
    printf("[Timer] %s: %.2lf ms, count=%d, avg = %.2lf ms\n", timerName, t[target], cnt[target], t[target]/cnt[target]);
}

void setTimer(const char* timerName )
{
    // Find exist timer
    int i;

    for( i = 0 ; i < g_num_timer; ++i ) {
	if( !strcmp(timer_alias[i], timerName) )
	    break;
    }

    if( i == g_num_timer ) {
	printf("error: cannot find the timer \"%s\"\n", timerName);
	return;
    }

    int target = i;
    t[target] = 0;
    cnt[target] = 0;
}
