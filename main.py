from random import *
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk, messagebox
from tkinter import *

class AnneallingSimulation:
    def __init__(self, cities, start, temperature, cooling, stopping):
        self.__cities = cities
        self.__initial_temperature = temperature
        self.__cooling_rate = cooling
        self.__stopping_temperature = stopping
        self.__start_city = start

    def distance(self, city1, city2):
        return np.linalg.norm(city1 - city2)

    def total_distance(self, tour, cities):
        distance = 0
        for i in range(len(tour) - 1):
            distance += np.linalg.norm(cities[tour[i]] - cities[tour[i+1]])
        distance += np.linalg.norm(cities[tour[-1]] - cities[tour[0]])
        return distance

    def acceptance_probability(self, old_cost, new_cost, temperature):
        if new_cost < old_cost:
            return 1.0
        return math.exp((old_cost - new_cost) / temperature)

    def simulated_annealing(self):
        num_cities = len(self.__cities)
        current_tour = [i for i in range(num_cities)]
        shuffle(current_tour)
        current_cost = self.total_distance(current_tour, self.__cities)
        temperature = self.__initial_temperature

        while temperature > self.__stopping_temperature:
            new_tour = current_tour.copy()
            i, j = sorted(sample(range(num_cities), 2))
            new_tour[i:j+1] = reversed(current_tour[i:j+1])        
            new_cost = self.total_distance(new_tour, self.__cities)

            if self.acceptance_probability(current_cost, new_cost, temperature) > random():
                current_tour = new_tour
                current_cost = new_cost

            temperature *= 1 - self.__cooling_rate

        current_tour = [self.__start_city] + current_tour + [self.__start_city]
        for i in range(num_cities):
            if (i != 0) and (i != num_cities) and (current_tour[i] == self.__start_city):
                current_tour.pop(i)
        current_cost = self.total_distance(current_tour, self.__cities)

        return current_tour, round(current_cost, 0)

class GeneticSimulation:
    def __init__(self, cities, start, Z, K, Pk, Pm, N):
        self.__cities = cities
        self.__start_city = start
        self.__Z = Z
        self.__K = K
        self.__Pcross = Pk
        self.__Pmut = Pm
        self.__N = N

    def FIND_WAY(self, ways, start, end = False):
        temp_ways = ways.copy()
        mx, my = temp_ways.shape
        new_start = start
        path = [new_start]

        there_no_queen, there_no_queen[new_start] = np.zeros(mx, dtype=bool), True  
        while not end:
            new_da_way = np.where((temp_ways[new_start] != 0) & (~there_no_queen))[0]  
            if new_da_way.size == 0:  
                break
            new_da_way = np.random.choice(new_da_way) 
            temp_ways[new_start, new_da_way], temp_ways[new_da_way, new_start] = 0, 0
            path.append(new_da_way)
            new_start = new_da_way
            there_no_queen[new_da_way] = True  
            end = np.all(temp_ways[new_start] == 0)
        path.append(start)

        return path

    def sum_WAY(self, ways, da_way):
        sum_da_way = 0
        for i in range(len(da_way) - 1):
            dar_way = da_way[i]
            dac_way = da_way[i+1]
            sum_da_way += ways[dar_way][dac_way]
        return (da_way, sum_da_way)

    def WAY_crossover(self, way1, way2, start, n):
        da_way_part = randint(2, n-1)
        p = way1[:da_way_part]
        if type(p) != list:
            p = p.tolist()
        for i in range(da_way_part, len(way2)):
            if way2[i] not in p:
                p.append(way2[i])
            else:
                p.append(-1)
        p[-1] = start
        for i in range(len(p)):
            if p[i] == -1:
                for j in range(0, n):
                    if j not in p:
                        p[i] = j
                        break
        p = np.array(p)
        return p

    def WAY_mutation(self, p, n):
        inds = sample(range(1, n), 2)
        P = np.array(p)
        P[inds[0]], P[inds[1]] = P[inds[1]], P[inds[0]]
        return P

    def simulated_genetic(self):
        generation = []
        for i in range(self.__K):
            g = self.FIND_WAY(self.__cities, self.__start_city)
            generation.append(self.sum_WAY(self.__cities, g))

        all_generations = [generation]
        best_sum = min([i[1] for i in generation])
        repeat = 1

        while repeat != self.__Z:
            next_generation = []
            for i in range(len(generation)):
                if randint(0, 100) < self.__Pcross:
                    next_generation.append(generation[i])
                else:
                    partner_id = i
                    while partner_id == i:
                        partner_id = randint(0, (len(generation)-1))

                    child1 = self.WAY_crossover(generation[i][0], generation[partner_id][0], self.__start_city, self.__N)
                    child2 = self.WAY_crossover(generation[partner_id][0], generation[i][0], self.__start_city, self.__N)

                    if randint(0, 100) < self.__Pmut:
                        child1_mut = self.WAY_mutation(child1, self.__N)
                        child2_mut = self.WAY_mutation(child2, self.__N)

                        P1 = self.sum_WAY(self.__cities, child1_mut)
                        P2 = self.sum_WAY(self.__cities, child2_mut)
                    else:
                        P1 = self.sum_WAY(self.__cities, child1)
                        P2 = self.sum_WAY(self.__cities, child2)

                    contest = [generation[i][1], P1[1], P2[1]]
                    win = min(contest)
                    win_id = contest.index(win)

                    if win_id == 0:
                        next_generation.append(generation[i])
                    elif win_id == 1:
                        next_generation.append(P1)
                    elif win_id == 2:
                        next_generation.append(P2)

            new_best_sum = min([i[1] for i in next_generation])
            if new_best_sum == best_sum:
                repeat += 1
            elif new_best_sum < best_sum:
                best_sum = new_best_sum
                repeat = 1

            all_generations.append(next_generation)
            generation = next_generation

        path = next(i for i in all_generations[-1] if i[1] == best_sum)

        return path[0], best_sum

class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.place()

    def quit_programm(self):
        if messagebox.askokcancel('Выход', 'Действительно хотите закрыть окно?'):
            quit()

    def create_graf(self, tour, color, N):
        edges = [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)]
        G = nx.complete_graph(N)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=12)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=2) 
        plt.show()

    def main_func(self):
        try:
            N = int(self.N_input.get())
            T1 = int(self.T1_input.get())
            T2 = int(self.T2_input.get())
            temperature = int(self.temperature_input.get())
            cooling = float(self.cooling_input.get())
            stopping = float(self.stopping_input.get())
            start_city = int(self.Start_city_input.get())
            Z = int(self.Z_input.get())
            K = int(self.K_input.get())
            Pmut = int(self.Pm_input.get())
            Pcross = int(self.Pc_input.get())

            cities = np.tril(np.random.randint(T1, T2, size=(N, N)), -1)
            matrix  = cities + cities.T

            matrix_output_title = ttk.Label(text="Матрица городов: ")
            matrix_output_title.place(x=350, y=10)
            rows_y = 10
            for i in range(N):
                matrix_row = ttk.Label(text=f"{matrix[i]}")
                matrix_row.place(x=450, y=rows_y)
                rows_y += 25

            annealing = AnneallingSimulation(matrix, start_city, temperature, cooling, stopping)
            tour, cost = annealing.simulated_annealing()

            genetic = GeneticSimulation(matrix, start_city, Z, K, Pcross, Pmut, N)
            tour2, cost2 = genetic.simulated_genetic()

            result_title = ttk.Label(text="Алгоритм отжига", font=("algerian", 15))
            result_title.place(x=15, y=320)
            result_output_tour = ttk.Label(text=f"Кратчайший путь: {tour}")
            result_output_tour.place(x=25, y=345)
            result_output_cost = ttk.Label(text=f"Длина пути: {cost}")
            result_output_cost.place(x=25, y=365)
            btn_annealing_show = ttk.Button(text="Показать граф", command=lambda: self.create_graf(tour, 'red', N))
            btn_annealing_show.place(x=15, y=385)

            result_title = ttk.Label(text="Генетический алгоритм", font=("algerian", 15))
            result_title.place(x=300, y=320)
            result_output_tour2 = ttk.Label(text=f"Кратчайший путь: {tour2}")
            result_output_tour2.place(x=315, y=345)
            result_output_cost2 = ttk.Label(text=f"Длина пути: {cost2}")
            result_output_cost2.place(x=315, y=365)
            btn_genetic_show = ttk.Button(text="Показать граф", command=lambda: self.create_graf(tour2, 'green', N))
            btn_genetic_show.place(x=300, y=385)

        except ValueError:
            messagebox.showerror("Ошибка!", "Пожалуйста, проверьте корректность введённых данных.")
            return 0    
        except:
            messagebox.showerror("Ошибка!", "Произошла ошибка во время оптимизации задачи.")
            return 0
            

    def main_window(self):
        title_start = ttk.Label(text="Введите следующие данные:", font=("algerian", 15))
        title_start.place(x=5, y=10)

        N_label = ttk.Label(text="N:")
        N_label.place(x=15, y=40)
        T1_label = ttk.Label(text="Нижняя граница:")
        T1_label.place(x=15, y=60)
        T2_label = ttk.Label(text="Верхняя граница:")
        T2_label.place(x=15, y=80)
        temperature_label = ttk.Label(text="Начальная температура:")
        temperature_label.place(x=15, y=100)
        cooling_label = ttk.Label(text="Скорость охлаждения:")
        cooling_label.place(x=15, y=120)
        stopping_label = ttk.Label(text="Температура остановки:")
        stopping_label.place(x=15, y=140)
        Start_city_label = ttk.Label(text="Начальный город:")
        Start_city_label.place(x=15, y=160)
        Z_label = ttk.Label(text="Кол-во повторов лучшей особи:")
        Z_label.place(x=15, y=180)
        K_label = ttk.Label(text="Кол-во особей в поколении:")
        K_label.place(x=15, y=200)
        Pm_label = ttk.Label(text="Вероятность мутации особи:")
        Pm_label.place(x=15, y=220)
        Pc_label = ttk.Label(text="Вероятность кроссовера между особями:")
        Pc_label.place(x=15, y=240)

        self.N_input = ttk.Entry()
        self.N_input.place(x=35, y=43, width=30, height=15)
        self.T1_input = ttk.Entry()
        self.T1_input.place(x=120, y=63, width=30, height=15)
        self.T2_input = ttk.Entry()
        self.T2_input.place(x=120, y=83, width=30, height=15)
        self.temperature_input = ttk.Entry()
        self.temperature_input.place(x=160, y=103, width=50, height=15)
        self.cooling_input = ttk.Entry()
        self.cooling_input.place(x=150, y=123, width=50, height=15)
        self.stopping_input = ttk.Entry()
        self.stopping_input.place(x=160, y=143, width=50, height=15)
        self.Start_city_input = ttk.Entry()
        self.Start_city_input.place(x=125, y=163, width=30, height=15)
        self.Z_input = ttk.Entry()
        self.Z_input.place(x=205, y=183, width=30, height=15)
        self.K_input = ttk.Entry()
        self.K_input.place(x=183, y=203, width=30, height=15)
        self.Pm_input = ttk.Entry()
        self.Pm_input.place(x=183, y=223, width=30, height=15)
        self.Pc_input = ttk.Entry()
        self.Pc_input.place(x=253, y=243, width=30, height=15)

        btn_start = ttk.Button(text="Старт", command=self.main_func)
        btn_start.place(x=15, y=265)


if __name__ == "__main__":
    window = Tk()
    window.geometry('%dx%d+%d+%d' % (800, 600, (window.winfo_screenwidth()/2) - (800/2), (window.winfo_screenheight()/2) - (600/2)))
    window.title("Алгоритмы оптимизации")
    app = App(window)
    window.protocol('WM_DELETE_WINDOW', app.quit_programm)
    app.main_window()
    app.mainloop()



