import numpy as np


def ACO_MV_R(pop_x, len_r, kesi, cum_p,  dn_r, up_r, K):
    n_z = np.zeros([1, len_r])

    #根据概率p轮盘赌选择一个个体
    idx_gr = RW_Select(cum_p)

    z, B, x0 = ROT(pop_x, idx_gr, len_r)


    for j in range(0, len_r):
        mu = z[idx_gr, j]
        sigma = kesi * np.sum(abs(mu - z[:, j])) / (K - 1)

        n_z[0, j] = mu + sigma * np.random.randn(1)
    x_r = np.dot(n_z, B.T) + x0
    x_r = Repair(x_r, dn_r,up_r)
    return x_r


def ROT(x_r, idx_gr, len_r):
    flag = (np.sum(np.sum(x_r - x_r[idx_gr, :])) != 0) & (len_r > 1)
    if flag:
        B = VCH(x_r, x_r[idx_gr, :])
    else:
        B = np.eye(len_r)

    if np.linalg.matrix_rank(B) != len_r:
        B = np.eye(len_r)

    z_r = np.dot(x_r - x_r[idx_gr, :], B)
    x0 = x_r[idx_gr, :]
    return z_r, B, x0


# 生成旋转矩阵
def VCH(s, s1):
    n = np.size(s, 1)
    A = np.zeros([n, n])
    for i in range(0, n):
        ds = np.sqrt(np.sum((s1[i:n] - s[:, i:n]) ** 2, 1))
        p = (ds ** 4) / np.sum(ds ** 4)
        idx1 = RW_Select(p)

        A[i, :] = s1 - s[idx1, :]
        s = np.delete(s, idx1, axis=0)

    if np.max(A) < 1e-5:
        B, non = np.linalg.qr(np.random.random([n, n]))
    else:
        B, non = np.linalg.qr(A.T)
    return B

def Repair(x_r, dn_r, up_r):
    for i in range(np.size(x_r, 0)):
        for j in range(np.size(x_r, 1)):
            if x_r[i,j] < dn_r or x_r[i,j] > up_r:
                x_r[i, j] = dn_r + np.random.rand(1)*(up_r - dn_r)

    return x_r

def RW_Select(p_set):
    lvalue = np.random.rand(1)
    probability = 0
    for i in range(np.size(p_set, 0)):
        probability += p_set[i]
        if probability >= lvalue:
                idx2 = i
                break
    return idx2

#################################
#     函数：ACO_MV离散算子       #
#################################
def ACO_MV_C(pop_x, len_c, w, q, N_lst, v_dv):
    x_c = np.zeros([1, len_c])
    for j in range(0, len_c):
        pl = Cal_pl(pop_x[:, j], N_lst[j], v_dv[j], w, q)
        idx_gc = RW_Select(pl)
        x_c[0, j] = v_dv[j, idx_gc]
    x_c = x_c.astype(float)
    return x_c

# 更新类别变量集合中每个元素可能被选择的概率
def Cal_pl(x_c, l, v_dv, w, q):
    u = np.zeros(l)
    wjl = np.zeros(l)
    wl = np.zeros(l)
    for i in range(0, l):
        idx_l = (x_c == v_dv[i])
        u[i] = np.sum(idx_l)

        if np.sum(idx_l) == 0:
            wjl[i] = 0
        else:
            wjl[i] = np.max(w[idx_l])

    eta = 100 * np.sum(u == 0)
    for i in range(0, l):
        if (eta > 0) & (u[i] > 0):
            wl[i] = wjl[i] / u[i] + q / eta
        elif (eta == 0) & (u[i] > 0):
            wl[i] = wjl[i] / u[i]
        elif (eta > 0) & (u[i] == 0):
            wl[i] = q / eta

    out = wl / np.sum(wl)
    return out

#################################
def ACO_MV_generates(K , M , database, len_r, len_c, dn_r, up_r, N_lst, v_dv):
    # K = 100
    # M = 100
    pop_x = database[0][:K]
    pop_y = database[1][:K]
    q = 0.05099 # Influene of the best-quality solutions in ACOmv
    kesi= 0.6795 # Width of the search in ACOmv
    x_r_generate = np.zeros((M,len_r))
    x_c_generate = np.zeros((M,len_c))
    w = np.zeros(K)
    p = np.zeros(K)
    cum_p = np.zeros(K)
    for j in range(0,K):
        pop_rank = j + 1
        w[j] = (1 / (q * K * np.pi)) \
            * np.exp(-((pop_rank - 1) ** 2) / (2 * (q * K) ** 2))  # the original paper is sqrt(2)

    for j in range(0,K):
        p[j] = w[j] / np.sum(w)
        cum_p[j] = np.sum(p[:j])

    for i in range(0,M):
        # pop_rank = i+1
        x_r_generate[i, :] = ACO_MV_R(pop_x[:,:len_r], len_r, kesi, cum_p, dn_r, up_r, K)
        x_c_generate[i, :] = ACO_MV_C(pop_x[:, len_r:], len_c,  w, q, N_lst, v_dv)

    return x_r_generate, x_c_generate
#################################