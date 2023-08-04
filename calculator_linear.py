import numpy as np


# 获取数据函数
def get_inp(num, x, index):
    p = []
    for i in range(num):
        p.append(float(x[i][index]))
    return p


# 假设函数
def hypothesis(num, inp, theta):
    result = float(theta[0])
    for index in range(num):
        result += inp[index] * float(theta[index + 1])
    return result


#代价函数
def cost_function(m, n, p, x, y):
    result = 0
    for i in range(m):
        result += (hypothesis(n, get_inp(n, x, i), p) - y[i])*(hypothesis(n, get_inp(n, x, i), p) - y[i])
    result /= 2 * m
    return result


#梯度求参
def gradient(m, n, a, x, y, p):
    count = 0
    while count < 10000:
        ans = []
        for j in range(n + 1):
            temp = 0
            if j == 0:
                for i in range(m):
                    temp += hypothesis(n, get_inp(n, x, i), p) - y[i]
                ans.append(temp)
            else:
                for i in range(m):
                    temp += (hypothesis(n, get_inp(n, x, i), p) - y[i]) * x[j - 1][i]
                ans.append(temp)
        is_break = True
        for i in range(n + 1):
            if a * ans[i] / m > 0.001 or a * ans[i] / m < -0.001:
                is_break = False
        if is_break:
            break
        for i in range(n + 1):
            p[i] -= a * ans[i] / m
        count += 1
    return p, count


#正规方程法
def normal_equation(m, x, y):
    mat_x = np.mat(x).T
    mat_y = np.mat(y).T
    mat_x = np.insert(mat_x, 0, np.ones(m), axis=1)
    return np.array(np.linalg.pinv(mat_x.T * mat_x) * mat_x.T * mat_y).T.flatten()
