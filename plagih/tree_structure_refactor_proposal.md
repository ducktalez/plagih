# Refactoring-Vorschlag: NodeStructure vs Node

## Aktuelles Problem

1. **NodeStructure** enthält Logik, die eigentlich zu **Node** gehört:
   - `is_term()` → prüft auf `Terminal` (eine Node-Subklasse)
   - `is_operator()` → prüft auf `BaseOperator` (eine Node-Subklasse)
   - `export_tree()` → kennt `Boolean`, `Number`, `Symbol`

2. **Rückgabetypen sind inkonsistent:**
   - `get_childs()` → `List['NodeStructure']`, aber es sind immer `Node`-Objekte
   - Rekursive Methoden erwarten `Node`, bekommen aber `NodeStructure`

3. **Zirkuläre Abhängigkeiten:**
   - `NodeStructure` referenziert `Node` (für Kinder)
   - `Node` erbt von `NodeStructure`

---

## Lösung: Klare Schichten-Trennung

### Schicht 1: `NodeStructure` (reine Baumstruktur)

```python
from typing import TypeVar, Generic, List, Optional
from dataclasses import dataclass, field

T = TypeVar('T', bound='NodeStructure')

@dataclass
class NodeStructure(Generic[T]):
    """
    Reine Baumstruktur - kennt NUR:
    - Kinder (childs)
    - Eltern (parent_node)
    - Wurzel (root_node)
    - Tiefe (depth)
    - Fixiert-Flag (is_fix)
    
    Kennt NICHT: Terminal, Operator, SymPy, etc.
    """
    childs: List[T] = field(default_factory=list)
    parent_node: Optional[T] = None
    root_node: Optional[T] = None
    depth: Optional[int] = None
    is_fix: bool = False

    # === Reine Struktur-Methoden ===
    
    def get_childs(self) -> List[T]:
        return self.childs

    def set_childs(self, childs: List[T]) -> None:
        self.childs = childs
        for child in childs:
            child.parent_node = self

    def add_child(self, child: T) -> None:
        self.childs.append(child)
        child.parent_node = self

    def is_root(self) -> bool:
        return self.parent_node is None

    def is_leaf(self) -> bool:
        """Strukturell: Hat keine Kinder"""
        return len(self.childs) == 0

    def get_max_depth(self, current_depth: int = 0) -> int:
        if self.is_leaf():
            return current_depth
        return max(c.get_max_depth(current_depth + 1) for c in self.childs)

    def repair_depth(self, depth: int = 0) -> None:
        self.depth = depth
        for child in self.childs:
            child.repair_depth(depth + 1)

    def repair_links(self, parent: Optional[T] = None, root: Optional[T] = None) -> None:
        self.parent_node = parent
        self.root_node = root or self
        for child in self.childs:
            child.repair_links(parent=self, root=self.root_node)
```

### Schicht 2: `Node` (GP-spezifische Logik)

```python
class Node(NodeStructure['Node']):
    """
    GP-Knoten mit:
    - SymPy-Funktion (symfun)
    - NumPy-Funktion (np_fun)
    - Typ-Informationen (xtype)
    - Darstellung (showme, sy_str, etc.)
    """
    symfun: Optional[Callable] = None
    np_fun: Optional[Callable] = None
    showme: str = ""
    xtype: tuple = ()
    # ... etc.

    # === Node-spezifische Methoden ===
    
    def is_term(self) -> bool:
        """Semantisch: Ist ein Terminal (Blatt mit Wert)"""
        return isinstance(self, Terminal)

    def is_operator(self) -> bool:
        """Semantisch: Ist ein Operator"""
        return isinstance(self, BaseOperator)

    def get_sympy_expr(self) -> sympy.Basic:
        # ... SymPy-Logik
        pass

    def get_value(self) -> Any:
        """Nur für Terminals"""
        if not self.is_term():
            raise TypeError("get_value() only for Terminal nodes")
        return self.childs[0]
```

---

## Konkrete Änderungen

### 1. Methoden verschieben: NodeStructure → Node

Diese Methoden gehören in `Node`, nicht in `NodeStructure`:

```python
# VERSCHIEBEN nach Node:
- is_term()
- is_operator()
- is_term_and_symbol()
- get_typus()  # oder entfernen, isinstance() ist klarer
- has_childs()  # → is_leaf() in NodeStructure (invertiert)
- export_tree()
- str_as_list()
- get_expr_symlike()
- list_mutable_nodes()
- get_all_nodes_visualize()
```

### 2. Rückgabetypen korrigieren

```python
# In NodeStructure:
def get_childs(self) -> List['Node']:  # NICHT NodeStructure
    return self.childs

# Oder mit Generic:
class NodeStructure(Generic[T]):
    def get_childs(self) -> List[T]:
        return self.childs

class Node(NodeStructure['Node']):
    pass  # get_childs() gibt jetzt List[Node] zurück
```

### 3. TYPE_CHECKING für Forward References

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plagih.trees import Terminal, BaseOperator, Number, Symbol, Boolean
```

---

## Minimalinvasive Alternative (Quick-Fix)

Falls das große Refactoring zu aufwändig ist:

### Option A: Nur Rückgabetypen fixen

```python
class NodeStructure:
    def get_childs(self) -> List['Node']:  # Ändere NodeStructure → Node
        return self.childs
```

### Option B: Type-Alias

```python
# Am Anfang der Datei:
TreeNode = 'Node'  # Alias für Klarheit

class NodeStructure:
    childs: List[TreeNode]
    parent_node: Optional[TreeNode]
    
    def get_childs(self) -> List[TreeNode]:
        return self.childs
```

### Option C: Protocol für Typ-Checks

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class TerminalProtocol(Protocol):
    def get_value(self) -> Any: ...
    def is_term(self) -> bool: ...

# Verwendung:
def process_terminal(node: TerminalProtocol) -> float:
    return float(node.get_value())
```

---

## Empfehlung

**Kurzfristig:** Option A (Rückgabetypen fixen) - 5 Minuten Arbeit, löst 80% der Warnungen

**Mittelfristig:** Methoden schrittweise von NodeStructure nach Node verschieben

**Langfristig:** Generics nutzen für saubere Typisierung

---

## Beispiel: Schrittweise Migration

```python
# Schritt 1: NodeStructure bleibt, aber Typen werden korrigiert
class NodeStructure:
    childs: List['Node']  # War: List['NodeStructure']
    
    def get_childs(self) -> List['Node']:
        return self.childs
    
    # is_leaf() statt has_childs() (invertierte Logik, aber strukturell)
    def is_leaf(self) -> bool:
        return len(self.childs) == 0

# Schritt 2: Node bekommt die semantischen Methoden
class Node(NodeStructure):
    def is_term(self) -> bool:
        return isinstance(self, Terminal)
    
    def has_childs(self) -> bool:
        """Semantisch: Operator hat Kinder"""
        return not self.is_term()  # Delegiert an is_leaf() wäre auch ok
```

