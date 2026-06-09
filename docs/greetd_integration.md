# Integración conceptual con greetd y KDE Plasma

> **Estado:** documento conceptual. En esta fase del proyecto **no** se modifica
> el login real del sistema. La integración descrita aquí es el objetivo a
> validar primero en una máquina virtual.

Este documento explica cómo encajaría el componente de reconocimiento facial de
`rp_face_login` dentro del arranque gráfico de un sistema Linux con KDE Plasma,
usando [greetd](https://sr.ht/~kennylevinsen/greetd/) como gestor de login.

---

## 1. Qué es greetd dentro del proyecto

[greetd](https://wiki.archlinux.org/title/Greetd) es un **daemon de login mínimo
y agnóstico**: no trae interfaz propia ni asume qué quieres lanzar. Su única
tarea es arrancar un *greeter* (la pantalla/agente de login) y, a través de un
protocolo IPC basado en JSON (`greetd-ipc`), permitir que ese greeter:

1. solicite la autenticación de un usuario a **PAM**, y
2. lance la sesión correspondiente si la autenticación tiene éxito.

Dentro de este proyecto, greetd es la **pieza del sistema operativo** que
arranca nuestro greeter y, sobre todo, **el que conserva la autenticación real
vía PAM**. Nuestro código de visión por computadora **no** sustituye a greetd ni
a PAM: solo aporta una **identidad candidata** (`elioth`, `emmanuel` o `guest`).

Diferencia clave frente a SDDM (el gestor por defecto de KDE): SDDM es un gestor
monolítico con UI propia; greetd delega la UI en un greeter externo, lo que hace
mucho más sencillo insertar un paso de reconocimiento facial sin parchear el
gestor.

---

## 2. Qué hace el greeter personalizado

El *greeter personalizado* (`face-login-greeter`, prototipado en
`scripts/face-login-greeter.sh`, Prompt 16) es un script/proceso delgado que:

1. Ejecuta el flujo de **`login-sim`** del paquete:
   cámara (≈5 s) → ZIP temporal → inferencia softmax por frame → agregación
   temporal → política de decisión.
2. Obtiene un **`selected_user`** (`elioth` | `emmanuel` | `guest`).
3. Entrega ese resultado al **despacho de sesión** (`session.dispatcher`), que en
   esta fase opera en modo `dry-run` o `command` (ver §8 sobre la forma correcta
   de hacerlo en producción).
4. Registra la decisión y los logs (`logs/face-login.log`).

> Importante: el greeter **produce identidad**, no concede acceso por sí mismo.
> La concesión de acceso es responsabilidad de greetd + PAM (ver §7).

---

## 3. Flujo completo

```text
systemd
   │  (arranca el servicio greetd.service)
   ▼
greetd  ──lee──>  /etc/greetd/config.toml
   │  (lanza el greeter como usuario "greeter")
   ▼
face-login-greeter        (scripts/face-login-greeter.sh)
   │  invoca: python -m rp_face_login.cli login-sim ...
   ▼
modelo facial             (captura → inferencia → agregación → decisión)
   │  produce:
   ▼
selected_user ∈ { elioth, emmanuel, guest }
   │
   ▼
session dispatch          (rp_face_login.session.dispatcher)
   │  dry-run  → solo imprime qué sesión se lanzaría
   │  command  → ejecuta un comando local configurado
   ▼
(en producción) greetd + PAM autentican y crean la sesión KDE Plasma
```

Resumen en una línea, tal como pide el plan:

```text
systemd → greetd → face-login-greeter → modelo facial → selected_user → session dispatch
```

---

## 4. Archivos relevantes

| Archivo | Rol |
|---|---|
| `/etc/greetd/config.toml` | Configuración de greetd: qué greeter lanzar, con qué usuario, y la sesión inicial/por defecto. |
| `/etc/pam.d/greetd` | Pila PAM que greetd usa para **autenticar** y abrir sesión. Aquí vive la autenticación real del SO. |
| `/usr/share/wayland-sessions/plasma.desktop` | Definición de la sesión **Plasma (Wayland)** que el greeter puede ofrecer/lanzar. |
| `/usr/share/xsessions/plasma.desktop` | Definición de la sesión **Plasma (X11)**. |
| `scripts/face-login-greeter.sh` | (Proyecto) Greeter experimental que llama a `login-sim`. |
| `configs/default.yaml` → `session_dispatch` | (Proyecto) Mapeo `usuario -> comando` leído por el dispatcher. |

Notas:
- Greeters gráficos como **gtkgreet**, **ReGreet** o **qtgreet** descubren las
  sesiones disponibles automáticamente desde `/usr/share/wayland-sessions/` y
  `/usr/share/xsessions/`, por lo que normalmente no hay que listar sesiones a
  mano.
- El greeter se ejecuta como el usuario **`greeter`** (creado con `useradd -M -G video greeter`), que necesita acceso al grupo `video` para usar la cámara.

---

## 5. Ejemplo conceptual de `config.toml`

> Ejemplo **ilustrativo**. No lo copies tal cual a una máquina principal.

```toml
[terminal]
vt = 1

# Sesión por defecto = el greeter. Aquí se usa un compositor mínimo (cage)
# para alojar nuestro greeter facial dentro de un entorno Wayland.
[default_session]
command = "cage -s -- /usr/local/bin/face-login-greeter"
user = "greeter"

# (Opcional) Sesión inicial / auto-login SOLO para pruebas en VM.
# Se ejecuta una única vez tras el arranque; al cerrarla vuelve el greeter.
# [initial_session]
# command = "startplasma-wayland"
# user = "elioth"
```

Y la pila PAM (`/etc/pam.d/greetd`) **mantiene** la autenticación estándar del
sistema (ejemplo conceptual, no copiar a producción sin entender cada línea):

```text
#%PAM-1.0
auth       include      system-local-login
account    include      system-local-login
session    include      system-local-login
```

La idea es **no eliminar** estas líneas: el reconocimiento facial se añade como
un paso *adicional* (idealmente un módulo PAM, ver §8), nunca como un reemplazo
que se salte PAM.

---

## 6. Advertencias

- **Prueba primero en una máquina virtual.** Un error en `config.toml` o en la
  pila PAM puede dejarte **sin acceso gráfico**. Ver `docs/vm_test_protocol.md`
  (Prompt 16).
- **No reemplaces SDDM en la máquina principal sin un plan de rollback.**
  Mantén SDDM instalado y ten a mano un TTY (`Ctrl+Alt+F3`) y el comando para
  volver: `systemctl disable --now greetd && systemctl enable --now sddm`.
- **No hardcodees contraseñas** en scripts, configs ni en el repositorio. El
  dispatcher del proyecto está diseñado explícitamente para no manejar
  credenciales.
- **No te saltes PAM en un entorno real.** Saltarse PAM convierte el "login
  facial" en un control de acceso falso: cualquiera con una foto o ante un fallo
  del modelo podría entrar. El reconocimiento facial **complementa**, no
  sustituye, a la autenticación del SO.
- **Privacidad:** los rostros son datos biométricos. No subas datasets ni
  modelos al repositorio (`data/`, `models/` ya están en `.gitignore`).

---

## 7. Reconocimiento facial vs. autenticación del SO

Es la distinción conceptual más importante del proyecto:

- **El reconocimiento facial produce *identidad* (claim).** Nuestro pipeline
  responde a "¿quién parece ser esta persona?" con `elioth`, `emmanuel` o, si no
  hay suficiente confianza/margen/frames, `guest` (rechazo). Es una **hipótesis
  probabilística**, no una garantía.
- **La autenticación del SO pertenece a greetd/PAM.** Decidir si esa identidad
  *puede* abrir una sesión —y crear esa sesión con las credenciales y permisos
  correctos— es competencia de PAM, que es el subsistema de seguridad del SO.

Por eso la arquitectura mantiene ambas responsabilidades **desacopladas**: el
modelo aporta `selected_user`; PAM/greetd siguen siendo la autoridad de
autenticación. Esto evita el antipatrón de "el greeter elige el usuario y entra
sin autenticar".

---

## 8. ¿Hay mejores opciones para el despacho/integración?

Investigación de alternativas y su encaje con este proyecto:

### Opción A — Módulo PAM de reconocimiento facial (recomendada para producción)
Integrar el reconocimiento como un **módulo PAM** (estilo
[Howdy](https://github.com/boltgolt/howdy), que ofrece "Windows Hello para
Linux" vía `pam_howdy.so`). Es la forma **conceptualmente correcta**: el
reconocimiento facial se convierte en un factor `auth` dentro de la pila PAM
(`/etc/pam.d/greetd`, `sudo`, lockscreen, etc.), normalmente como
`auth sufficient` con *fallback* a contraseña.

- **Ventajas:** la autenticación sigue dentro de PAM (no se evita); funciona en
  todas partes (login, `sudo`, `su`, bloqueo de pantalla); respeta la separación
  identidad/autenticación de §7.
- **Inconvenientes/realidad actual:** la combinación **Howdy + greetd** tiene un
  bug conocido y abierto (`pam_setcred: PERM_DENIED`) por cómo greetd gestiona
  las credenciales de sesión; además muchos setups requieren "pulsar Enter"
  antes de que arranque la cámara (falta de PAM asíncrono en algunos
  greeters/DMs). Es viable, pero hoy frágil con greetd concretamente.

### Opción B — Greeter gráfico existente + IPC (pragmática)
Usar un greeter ya hecho que hable el protocolo `greetd-ipc`
([ReGreet](https://github.com/rharish101/ReGreet) GTK4, gtkgreet, qtgreet,
tuigreet) y añadir el paso facial **antes** de delegar en PAM. El greeter sigue
pidiendo a PAM la autenticación final.

- **Ventajas:** UI madura, descubrimiento automático de sesiones Plasma,
  integración estándar con greetd.
- **Inconvenientes:** hay que personalizar/extender el greeter; el "factor
  facial" queda fuera de PAM salvo que se combine con la Opción A.

### Opción C — Greeter propio que habla `greetd-ipc` (máximo control)
Escribir un greeter que implemente directamente el protocolo IPC de greetd:
ejecuta `login-sim`, obtiene `selected_user`, y luego usa la **secuencia IPC de
greetd** (`create_session` → `post_auth_message_response` → `start_session`)
para que **PAM** complete la autenticación.

- **Ventajas:** control total; el despacho de sesión se hace por el canal
  oficial de greetd en lugar de ejecutar comandos sueltos.
- **Inconvenientes:** más trabajo; hay que manejar bien el IPC y los casos de
  error.

### Recomendación para este proyecto
1. **Ahora (académico):** mantener el `SessionDispatcher` desacoplado en
   `dry-run`/`command`. Es seguro, testeable y demuestra la arquitectura sin
   tocar el SO.
2. **Siguiente paso real:** preferir la **Opción C** (greeter que usa el
   `greetd-ipc` para que PAM autentique) sobre ejecutar comandos crudos, porque
   delega la creación de sesión en el canal oficial y conserva PAM.
3. **Objetivo "production-grade":** mover el factor facial a un **módulo PAM**
   (Opción A) para cumplir plenamente la separación de §7, asumiendo y mitigando
   los problemas conocidos de Howdy+greetd (probar en VM, mantener fallback a
   contraseña, no usarlo como único factor).

En todos los casos se mantiene la regla de oro: **el reconocimiento facial
aporta identidad; PAM autentica.**

---

## Referencias

- greetd — Arch Wiki: <https://wiki.archlinux.org/title/Greetd>
- greetd(5) / greetd-ipc(7): <https://man.sr.ht/~kennylevinsen/greetd/>
- ReGreet: <https://github.com/rharish101/ReGreet>
- Howdy (PAM facial): <https://github.com/boltgolt/howdy>
- Howdy + greetd (issue `pam_setcred`): <https://github.com/boltgolt/howdy/issues/991>
