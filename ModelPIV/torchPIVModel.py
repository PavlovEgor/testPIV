from ModelPIV.BasicModelPIV import BasicModelPIV

from torchPIV import OfflinePIV

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve


class torchPIVModel(BasicModelPIV):

    def __init__(self, numOfPixelsX, numOfPixelsY):
        super().__init__(numOfPixelsX, numOfPixelsY)

        self.tmp_folder_name = "tmp"
        self.folder_mode = "pairs"
        self.file_fmt = "jpg"

        self.device = "cpu"

        self.wind_size = 64
        self.overlap = 32
        self.multipass = 2
        self.multipass_mode = "DWS"
        self.multipass_scale = 2.0

    def set_setting(self,
                    tmp_folder_name="tmp",
                    folder_mode="pairs",
                    file_fmt="jpg",
                    device="cpu",
                    wind_size=32,
                    overlap=16,
                    multipass=2,
                    multipass_mode="CWS",
                    multipass_scale=2.0
                    ):
        self.tmp_folder_name = tmp_folder_name
        self.folder_mode = folder_mode
        self.file_fmt = file_fmt

        self.device = device

        self.wind_size = wind_size
        self.overlap = overlap
        self.multipass = multipass
        self.multipass_mode = multipass_mode
        self.multipass_scale = multipass_scale

    def predict(self, particles):

        if particles.isEvolve:
            initial_picture = self.generatePicture(condition="initial")
            final_picture = self.generatePicture(condition="final")

            self.savePicture(initial_picture, 'a.jpg', self.tmp_folder_name)
            self.savePicture(final_picture, 'b.jpg', self.tmp_folder_name)

            piv_gen = OfflinePIV(
                folder=self.tmp_folder_name,  # Path to experiment
                device=self.device,  # Device name
                file_fmt=self.file_fmt,
                wind_size=self.wind_size,
                overlap=self.overlap,
                dt=particles.dt*1000,  # Time between frames, mcs
                scale=particles.X_scale/self.numOfPixelsX,  # mm/pix
                multipass=self.multipass,
                multipass_mode=self.multipass_mode,  # CWS or DWS
                multipass_scale=self.multipass_scale,  # Window downscale on each pass
                folder_mode=self.folder_mode  # Pairs or sequential frames
            )

            results = []
            for out in piv_gen():
                results.append(out)
            self.X, self.Y, self.Vx, self.Vy = results[0]

        else:
            print("particles is not evolved yet")
            exit()

    def error(self, flow, n=1):

        self.VxGround, self.VyGround = flow.velocity(self.X, self.Y)

        L = self.wind_size * self.particles.X_scale / self.numOfPixelsX

        for i in range(1, n+1):
            for j in range(1, n+1):
                VxGround_tmp, VyGround_tmp = flow.velocity(self.X - L/2 + j * L / (n+1), self.Y - L/2 + i * L / (n+1))

                self.VxGround += VxGround_tmp
                self.VyGround += VyGround_tmp

        self.VxGround /= n + 1
        self.VyGround /= n + 1

        self.VxGround = np.reshape(self.VxGround, self.Vx.shape)
        self.VyGround = np.reshape(self.VyGround, self.Vy.shape)

        self.VxGround = np.flipud(self.VxGround)
        self.VyGround = np.flipud(self.VyGround)

        self.VyGround *= -1

        return np.sqrt(np.mean((self.Vx - self.VxGround) ** 2) + np.mean((self.Vy - self.VyGround) ** 2))
    
    def correct(self):
        """
        Решает задачу проекции на соленоидальное поле с помощью множителей Лагранжа
        """
            
        n_rows, n_cols = self.Vx.shape
        n_points = n_rows * n_cols

        x = self.Vx
        y = self.Vy
        
        # Количество уравнений связи (внутренние точки)
        # Используем все точки, кроме граничных, которые определяются по условию
        n_constraints = n_rows * n_cols
        
        # Строим матрицу связей A размера n_constraints x (2*n_points)
        # Уравнения имеют вид: A * [u_flat; v_flat] = 0
        
        A = lil_matrix((n_constraints, 2 * n_points))
        
        for i in range(n_rows):
            for j in range(n_cols):
                constraint_idx = i * n_cols + j
                
                # Индексы для i-1, i+1 с граничными условиями (ближайший элемент)
                i_minus = max(0, i - 1)
                i_plus = min(n_rows - 1, i + 1)
                j_minus = max(0, j - 1)
                j_plus = min(n_cols - 1, j + 1)
                
                # Член u_{i,j+1} - u_{i,j-1}
                if j_plus != j_minus:  # Если соседи разные
                    # u_{i,j+1}
                    u_idx_plus = i * n_cols + j_plus
                    A[constraint_idx, u_idx_plus] += 1.0
                    
                    # u_{i,j-1}
                    u_idx_minus = i * n_cols + j_minus
                    A[constraint_idx, u_idx_minus] -= 1.0
                
                # Член v_{i+1,j} - v_{i-1,j}
                if i_plus != i_minus:  # Если соседи разные
                    # v_{i+1,j}
                    v_idx_plus = n_points + i_plus * n_cols + j
                    A[constraint_idx, v_idx_plus] += 1.0
                    
                    # v_{i-1,j}
                    v_idx_minus = n_points + i_minus * n_cols + j
                    A[constraint_idx, v_idx_minus] -= 1.0
        
        A = csr_matrix(A)
        
        # Формируем правую часть для задачи минимизации
        # Минимизируем ||u-x||^2 + ||v-y||^2 при Au = 0
        # Лагранжиан: L = 0.5*(||u-x||^2 + ||v-y||^2) + λ^T A [u; v]
        # Условия стационарности:
        # u - x + A_u^T λ = 0  -> u = x - A_u^T λ
        # v - y + A_v^T λ = 0  -> v = y - A_v^T λ
        # A [u; v] = 0
        
        # Подставляем: A [x; y] - A A^T λ = 0
        # => A A^T λ = A [x; y]
        
        # Вектор известных значений
        b_known = np.zeros(2 * n_points)
        b_known[:n_points] = x.flatten()
        b_known[n_points:] = y.flatten()
        
        # Правая часть для λ
        rhs_lambda = A @ b_known
        
        # Решаем систему A A^T λ = rhs_lambda
        AAT = A @ A.T
        AAT = csr_matrix(AAT)
        
        # Добавляем малую регуляризацию для устойчивости
        reg = 1e-10
        AAT_reg = AAT + reg * eye(n_constraints, format='csr')
        
        try:
            lambda_sol = spsolve(AAT_reg, rhs_lambda)
        except Exception as e:
            print(f"Ошибка при решении: {e}")
            # Альтернативный метод: наименьшие квадраты
            from scipy.sparse.linalg import lsmr
            lambda_sol = lsmr(AAT_reg, rhs_lambda)[0]
        
        # Вычисляем u и v
        u_flat = b_known[:n_points] - (A[:, :n_points].T @ lambda_sol)
        v_flat = b_known[n_points:] - (A[:, n_points:].T @ lambda_sol)
        
        # Преобразуем обратно в 2D массивы
        u = u_flat.reshape(n_rows, n_cols)
        v = v_flat.reshape(n_rows, n_cols)
        
        self.Vx = u
        self.Vy = v
