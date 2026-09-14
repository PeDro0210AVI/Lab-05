"""Modulo de funciones reutilizables para interactuar con entornos de Gymnasium / ALE.

Proporciona: crear_entorno, agente_aleatorio, agente_regla_simple, ejecutar_episodio
y generar_video_agente. Pensado para funcionar tanto con ALE/SpaceInvaders-v5 como con
cualquier otro entorno de Gymnasium con espacio de accion discreto.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import gymnasium as gym

# Los binarios de ROM (licenciados para uso libre/investigacion por Farama desde ALE 0.8.0)
# se distribuyen en assets/roms/ dentro del repositorio; ALE los busca via ALE_ROMS_DIR.
_ROMS_DIR = Path(__file__).resolve().parent.parent / "assets" / "roms"
os.environ.setdefault("ALE_ROMS_DIR", str(_ROMS_DIR))

import ale_py  # noqa: E402  (debe importarse despues de fijar ALE_ROMS_DIR)

gym.register_envs(ale_py)


def crear_entorno(nombre_entorno, video_folder=None, episode_trigger=None, name_prefix="rl-video", **kwargs):
    """Crea y retorna un entorno de Gymnasium (funciona con Space Invaders o cualquier otro).

    Si se especifica `video_folder`, el entorno se crea con render_mode="rgb_array" y se
    envuelve con gymnasium.wrappers.RecordVideo, grabando los episodios indicados por
    `episode_trigger` (por defecto, todos los episodios).
    """
    if video_folder is not None:
        kwargs.setdefault("render_mode", "rgb_array")

    env = gym.make(nombre_entorno, **kwargs)

    if video_folder is not None:
        os.makedirs(video_folder, exist_ok=True)
        if episode_trigger is None:
            episode_trigger = lambda episode_id: True  # noqa: E731
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            name_prefix=name_prefix,
        )

    return env


def agente_aleatorio(observation, env):
    """Agente baseline: retorna una accion muestreada uniformemente de env.action_space."""
    return env.action_space.sample()


def agente_regla_simple(observation, env):
    """Agente basado en una regla fija (sin entrenamiento).

    Para entornos ALE con acciones RIGHTFIRE/LEFTFIRE (p. ej. Space Invaders), alterna
    bloques de 10 pasos moviendose y disparando hacia la derecha y hacia la izquierda.
    Para cualquier otro entorno de Gymnasium, cae de vuelta al muestreo aleatorio.
    """
    if not hasattr(agente_regla_simple, "_contador"):
        agente_regla_simple._contador = 0
    contador = agente_regla_simple._contador
    agente_regla_simple._contador += 1

    meanings = getattr(env.unwrapped, "get_action_meanings", None)
    meanings = meanings() if meanings is not None else None

    if meanings and "RIGHTFIRE" in meanings and "LEFTFIRE" in meanings:
        accion_nombre = "RIGHTFIRE" if (contador // 10) % 2 == 0 else "LEFTFIRE"
        return meanings.index(accion_nombre)

    return env.action_space.sample()


def ejecutar_episodio(env, funcion_agente, max_steps=10000):
    """Ejecuta un episodio completo con `funcion_agente` hasta terminated/truncated o max_steps.

    Retorna (steps, total_reward): pasos ejecutados y recompensa (return) acumulada.
    """
    observation, info = env.reset()
    terminated = truncated = False
    steps = 0
    total_reward = 0.0

    while not (terminated or truncated) and steps < max_steps:
        action = funcion_agente(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += reward

    return steps, total_reward


def generar_video_agente(nombre_entorno, funcion_agente, video_folder, name_prefix, n_episodios=1):
    """Crea el entorno con grabacion de video, ejecuta n_episodios y cierra el entorno.

    Retorna (video_paths, metricas): rutas .mp4 generadas (ordenadas) y una lista de
    dicts {"episodio", "steps", "return"} con las metricas de cada episodio.
    """
    env = crear_entorno(
        nombre_entorno,
        video_folder=video_folder,
        episode_trigger=lambda episode_id: True,
        name_prefix=name_prefix,
    )

    metricas = []
    for episodio in range(n_episodios):
        steps, total_reward = ejecutar_episodio(env, funcion_agente)
        metricas.append({"episodio": episodio, "steps": steps, "return": total_reward})

    env.close()  # indispensable para que moviepy/RecordVideo escriba los .mp4 a disco

    video_paths = sorted(glob.glob(os.path.join(video_folder, f"{name_prefix}*.mp4")))
    return video_paths, metricas
