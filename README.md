# gabriel_nlopt

Trabalho de implementação de algoritmos de otimização não-linear para a disciplina de Otimização (COS360) de 2022/2 na Universidade Federal do Rio de Janeiro.

## Algoritmos implementados

**Obs:** todos os algoritmos utilizam a regra de armijo para atualização do passo.

- [x] Método do gradiente
- [x] Método de Newton
- [x] Método de Quasi-Newton (BFGS)

## Como usar

### Instalação

#### Requisitos

- Python 3.9+ (testado com 3.9.15)
- [Poetry](https://python-poetry.org/) (testado com 1.2.1)

#### Passos

- Clone esse repositório
- Instale as dependências e a biblioteca com `poetry install`

### Execução

Para minimizar uma função, você deve usar o template abaixo:

```py
from gabriel_nlopt import Function, minimize
import numpy as np


class SuaFuncaoObjetivo(Function):
    def __call__(self, x):
        # Implemente aqui sua função objetivo
        # Por exemplo: x1^2 + x2^2
        return (np.power(x[0], 2) + np.power(x[1], 2)).astype(np.float64)

    def gradient(self, x):
        # Implemente aqui o gradiente da sua função objetivo
        # Por exemplo: [2*x1, 2*x2]
        return np.array([2 * x[0], 2 * x[1]], dtype=np.float64)

    def hessian(self, x):
        # Implemente aqui a hessiana da sua função objetivo
        # Por exemplo: [[2, 0], [0, 2]]
        return np.array([[2, 0], [0, 2]], dtype=np.float64)


if __name__ == "__main__":
    # Defina o ponto inicial e instancie a sua função objetivo
    x0 = np.array([5, 5], dtype=np.float64)
    sua_funcao_objetivo = SuaFuncaoObjetivo()

    # Minimize a sua função objetivo usando um dos métodos disponíveis
    x_min = minimize(
        sua_funcao_objetivo, x0, method="newton"
    )  # ou "gradient" ou "bfgs"
    print(
        f"O melhor resultado é o ponto {x_min}, com valor {sua_funcao_objetivo(x_min)}"
    )
```
