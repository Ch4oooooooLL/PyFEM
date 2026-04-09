class Material:
    """
    表示材料属性。
    """
    def __init__(self, E: float, rho: float, nu: float = 0.3):
        """
        :param E: 弹性模量 (Elastic Modulus)
        :param rho: 密度 (Density)
        :param nu: 泊松比 (Poisson's ratio)
        """
        self.E: float = E
        self.rho: float = rho
        self.nu: float = nu

    def __repr__(self) -> str:
        return f"Material(E={self.E}, rho={self.rho}, nu={self.nu})"
