import torch


def einsum():
    N = torch.randn(1, 200, 15)
    q_cps = torch.randn(10, 15, 7)

    q1 = torch.einsum('ijk,lkm->ljm', N, q_cps)
    q2 = torch.matmul(N, q_cps)

    print(q1)
    print(q2)

    print((q1 == q2).all())

    t_cps = torch.randn(10, 15, 1)

    dtau_dt1 = torch.einsum('ijk,lkm->ljm', N, t_cps)
    dtau_dt2 = torch.matmul(N, t_cps)

    print(dtau_dt1)
    print(dtau_dt2)

    print((dtau_dt1 == dtau_dt2).all())


if __name__ == '__main__':
    einsum()
