## Q1. This question was asked by: Facebook. Let's say you're playing a dice game. You have 2 die. 
### 1. What's the probability of rolling at least one 3?
Look at it this way, the probability of rolling at least one 3, means we can roll 3 once, or roll 3 twice. And bear in mind, there are 6*6 == 36 total combinations
P(at least one 3) = P(getting one 3) + P(getting two 3s)
                  = (10/36) + (1/36)
                  = 11/36
                  
### 2. What's the probability of rolling at least one 3 given N die?
If we think about it, when we have 3 dices, there will be a lot of combinations to consider. How about we structure our solution in a different way by finding the 
complement, i.e. probablility of not getting a 3 at all?

Assuming we have 2 dice, the probabolity of not getting a 3 at all would be (5/6)^2
Assuming we have 3 dice, the probabolity of not getting a 3 at all would be (5/6)^3

Thus, we can form the equation for the probability as: 1 - (5/6)^N

## Q2. This question was asked by: Fitbit
## Given a list of numbers and window_size (int), calculate the moving window average.

def moving_avg(arr, val):
    i = 0; mov_avg = []
    while i < (len(arr) - val + 1):
        avg_val = sum(arr[i:i+val])/val
        mov_avg.append(avg_val)
        i += 1
    return mov_avg

input1 = [1,2,3,4,5,6]
val = 5
print(moving_avg(input1, val))

## Q3. What do you think the distribution of time spent per day on Facebook looks like? What metrics would you use to describe that distribution?
For Facebook, the first intuition might be it's normal distribution, but I think it depends. We can look at this as a problem of probablities. Assuming a user who is currently browsing FB has probability of *p* to leave FB after a certain unit of time, denoted as *t*. *p* could be described as a function of *t*, i.e. the longer the user is on FB, the lower/ higher the *p* for the user to stop using FB. Time spend per day in Facebook also depends on the period of the day where most posts are being posted/ videos posted/ more contents in general.




